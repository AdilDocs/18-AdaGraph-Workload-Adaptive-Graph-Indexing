"""Phase 1: static baseline on the testbed.

Builds the base Vamana graph, generates query workloads and measures, for each workload,
Recall@10 against the number of distance computations per query (the cost of Eq. (2)).
The gap between in-distribution and out-of-distribution queries is the motivation for
AdaGraph; Phase 2 will try to close it with adaptive edges.

Examples (run from the folder that contains ./data):
    python run_phase1.py --dataset synthetic
    python run_phase1.py --dataset sift --n 200000
    python run_phase1.py --dataset t2i  --n 200000
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import numba

from adagraph.data import load_dataset, cached_groundtruth, recall_at_k
from adagraph.graph import VamanaGraph
from adagraph.workload import QueryStreams

L_SWEEP = [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "sift", "t2i"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200000, help="number of base vectors to index")
    ap.add_argument("--R", type=int, default=32, help="max out-degree of the base graph")
    ap.add_argument("--L-build", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--stream-len", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def interp_cost(recalls, costs, target):
    """Distance computations needed to reach a recall target (linear interpolation)."""
    r = np.asarray(recalls)
    c = np.asarray(costs, dtype=float)
    if r.max() < target:
        return float("nan")
    i = int(np.argmax(r >= target))
    if i == 0:
        return float(c[0])
    f = (target - r[i - 1]) / max(1e-12, r[i] - r[i - 1])
    return float(c[i - 1] + f * (c[i] - c[i - 1]))


def main():
    args = parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    n_arg = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    tag = "%s_n%d" % (args.dataset, n_arg)
    cache = os.path.join(args.data_dir, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print("== AdaGraph Phase 1 | dataset=%s n=%d threads=%d" % (args.dataset, n_arg, numba.get_num_threads()))

    # 1. data -------------------------------------------------------------------------
    t0 = time.time()
    base, pools = load_dataset(args.dataset, args.data_dir, n=n_arg)
    print("[data] base %s, pools: %s (%.1fs)" % (
        base.shape, ", ".join("%s=%d" % (k, v.shape[0]) for k, v in pools.items()), time.time() - t0))

    # 2. ground truth for the evaluation pools (not for 'hist', which is ground-truth-free)
    eval_pools = [p for p in ("id", "ood") if p in pools]
    gts = {}
    for p in eval_pools:
        t0 = time.time()
        gts[p] = cached_groundtruth(cache, p, base, pools[p], k=100)
        print("[gt] %s: %s (%.1fs)" % (p, gts[p].shape, time.time() - t0))

    # 3. base graph -------------------------------------------------------------------
    gpath = os.path.join(cache, "vamana_R%d_L%d_a%.2f_p%d.npz" % (args.R, args.L_build, args.alpha, args.passes))
    if os.path.exists(gpath):
        g = VamanaGraph.load(gpath, base)
        print("[graph] loaded %s (built in %.1fs)" % (gpath, g.build_time))
    else:
        print("[graph] building Vamana R=%d L=%d alpha=%.2f passes=%d ..." % (args.R, args.L_build, args.alpha, args.passes))
        g = VamanaGraph(base, R=args.R, L_build=args.L_build, alpha=args.alpha, passes=args.passes).build()
        g.save(gpath)
        print("[graph] built in %.1fs, saved to %s" % (g.build_time, gpath))
    print("[graph] avg out-degree %.1f, max %d, isolated nodes %d"
          % (g.deg.mean(), g.deg.max(), int((g.deg == 0).sum())))

    # 4. workloads --------------------------------------------------------------------
    qs = QueryStreams({p: pools[p] for p in eval_pools}, n_clusters=64)
    workloads = {}
    for p in eval_pools:
        workloads["%s-uniform" % p] = qs.uniform(p, args.stream_len, seed=1)
    main_pool = "ood" if "ood" in eval_pools else "id"
    workloads["%s-skewed" % main_pool] = qs.skewed(main_pool, args.stream_len, seed=2)
    pids, rows, _ = qs.drift(main_pool, args.stream_len, phases=5, seed=3)
    workloads["%s-drift" % main_pool] = (pids, rows)
    if "id" in eval_pools and "ood" in eval_pools:
        workloads["id-to-ood"] = qs.id_to_ood("id", "ood", args.stream_len, seed=4)

    # 5. sweep ------------------------------------------------------------------------
    g.search(base[:8], L=10, k=args.k)  # JIT warm-up
    rows_out = []
    summary = []
    for wname, (pids, rws) in workloads.items():
        queries = qs.gather(pids, rws)
        gt = np.empty((pids.shape[0], args.k), np.int32)
        for j, name in enumerate(qs.names):
            m = pids == j
            gt[m] = gts[name][rws[m], :args.k]
        print("\n[%s] %d queries" % (wname, queries.shape[0]))
        print("   L   recall@%d   dist/query   hops/query      QPS" % args.k)
        recs, costs = [], []
        for L in L_SWEEP:
            if L < args.k:
                continue
            t0 = time.time()
            ids, nd, hp = g.search(queries, L=L, k=args.k)
            dt = time.time() - t0
            rec = recall_at_k(ids, gt, args.k)
            recs.append(rec)
            costs.append(nd.mean())
            print("%4d   %8.4f   %10.1f   %10.1f   %8.0f" % (L, rec, nd.mean(), hp.mean(), queries.shape[0] / dt))
            rows_out.append([args.dataset, n_arg, wname, L, rec, nd.mean(), hp.mean(), queries.shape[0] / dt])
        summary.append((wname, interp_cost(recs, costs, 0.90), interp_cost(recs, costs, 0.95)))

    out_csv = os.path.join(args.out_dir, "phase1_%s.csv" % tag)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "workload", "L", "recall@%d" % args.k, "dist_per_query", "hops_per_query", "qps"])
        w.writerows(rows_out)

    print("\n== Summary: distance computations per query needed to reach a recall target")
    print("%-16s %14s %14s" % ("workload", "recall 0.90", "recall 0.95"))
    for wname, c90, c95 in summary:
        print("%-16s %14.0f %14.0f" % (wname, c90, c95))
    print("\nResults written to %s" % out_csv)


if __name__ == "__main__":
    main()
