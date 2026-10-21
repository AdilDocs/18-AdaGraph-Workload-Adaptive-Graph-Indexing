"""Phase 2: go/no-go test for adaptive edges.

Adds an adaptive edge layer built only from the search results of history queries
(no ground truth), then reruns the Phase 1 evaluation and compares the number of
distance computations needed to reach recall 0.90 / 0.95 with the base graph alone.

The history pool never overlaps the evaluation queries:
    t2i        history = text queries from the learn set (OOD); evaluated on public text queries
    sift       history = held-out SIFT base vectors (ID) -> checks that adaptation does no harm
    synthetic  history = OOD samples from the shifted clusters

Examples:
    python run_phase2.py --dataset t2i  --n 200000
    python run_phase2.py --dataset sift --n 200000
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
from adagraph.adaptive import AdaptiveLayer
from adagraph.workload import QueryStreams
from run_phase1 import L_SWEEP, interp_cost


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "sift", "t2i"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--L-build", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--stream-len", type=int, default=10000)
    ap.add_argument("--A", default="8,16", help="adaptive edge budgets per node to try")
    ap.add_argument("--K-obs", type=int, default=10, help="top results kept per history query")
    ap.add_argument("--L-obs", type=int, default=100, help="beam width for history queries")
    ap.add_argument("--n-hist", type=int, default=100000)
    ap.add_argument("--min-count", type=int, default=1, help="min number of history queries supporting an edge")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def sweep(search_fn, queries, gt, k):
    recs, costs = [], []
    for L in L_SWEEP:
        if L < k:
            continue
        ids, nd, _ = search_fn(queries, L, k)
        recs.append(recall_at_k(ids, gt, k))
        costs.append(float(nd.mean()))
    return recs, costs


def main():
    args = parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    n_arg = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    tag = "%s_n%d" % (args.dataset, n_arg)
    cache = os.path.join(args.data_dir, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print("== AdaGraph Phase 2 | dataset=%s n=%d threads=%d" % (args.dataset, n_arg, numba.get_num_threads()))

    base, pools = load_dataset(args.dataset, args.data_dir, n=n_arg)
    if "hist" not in pools:
        raise SystemExit("this dataset has no history pool")
    hist = pools["hist"][:args.n_hist]
    eval_pools = [p for p in ("id", "ood") if p in pools]
    gts = {p: cached_groundtruth(cache, p, base, pools[p], k=100) for p in eval_pools}

    gpath = os.path.join(cache, "vamana_R%d_L%d_a%.2f_p%d.npz" % (args.R, args.L_build, args.alpha, args.passes))
    if os.path.exists(gpath):
        g = VamanaGraph.load(gpath, base)
        print("[graph] loaded base graph (built in %.1fs)" % g.build_time)
    else:
        print("[graph] building base graph ...")
        g = VamanaGraph(base, R=args.R, L_build=args.L_build, alpha=args.alpha, passes=args.passes).build()
        g.save(gpath)

    # same workloads and seeds as Phase 1
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
    wdata = {}
    for wname, (pids, rws) in workloads.items():
        gt = np.empty((pids.shape[0], args.k), np.int32)
        for j, name in enumerate(qs.names):
            m = pids == j
            gt[m] = gts[name][rws[m], :args.k]
        wdata[wname] = (qs.gather(pids, rws), gt)

    # methods: base graph + one adaptive layer per budget A
    methods = {"base": lambda q, L, k: g.search(q, L, k)}
    info = {"base": "R=%d, %d edges" % (args.R, int(g.deg.sum()))}
    for A in [int(a) for a in args.A.split(",") if a.strip()]:
        print("[adaptive] A=%d K=%d L_obs=%d on %d history queries ..." % (A, args.K_obs, args.L_obs, hist.shape[0]))
        layer = AdaptiveLayer(g, A=A, K=args.K_obs, L_obs=args.L_obs, alpha=args.alpha,
                              min_count=args.min_count).fit(hist)
        name = "adaptive-A%d" % A
        methods[name] = (lambda lay: (lambda q, L, k: lay.search(q, L, k)))(layer)
        s = layer.stats
        info[name] = "+%d edges (+%.1f%%), fit %.1fs" % (s["adaptive_edges"], s["extra_memory_pct"],
                                                         s["observe_time_s"] + s["select_time_s"])

    g.search(base[:8], L=10, k=args.k)  # warm-up
    results = {}
    rows_out = []
    for wname, (queries, gt) in wdata.items():
        for mname, fn in methods.items():
            t0 = time.time()
            recs, costs = sweep(fn, queries, gt, args.k)
            c90, c95 = interp_cost(recs, costs, 0.90), interp_cost(recs, costs, 0.95)
            results[(wname, mname)] = (c90, c95)
            for L, r, c in zip([l for l in L_SWEEP if l >= args.k], recs, costs):
                rows_out.append([args.dataset, n_arg, wname, mname, L, r, c])
            print("  %-14s %-14s cost@0.90=%7.0f  cost@0.95=%7.0f  (%.1fs)" % (wname, mname, c90, c95, time.time() - t0))

    out_csv = os.path.join(args.out_dir, "phase2_%s.csv" % tag)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "workload", "method", "L", "recall@%d" % args.k, "dist_per_query"])
        w.writerows(rows_out)

    print("\n== Methods")
    for m, s in info.items():
        print("  %-14s %s" % (m, s))
    print("\n== Summary: distance computations per query to reach recall (lower is better);"
          " x = speedup over base")
    mnames = list(methods.keys())
    header = "%-14s" % "workload" + "".join("%22s" % m for m in mnames)
    for target_i, target in enumerate((0.90, 0.95)):
        print("\nrecall %.2f" % target)
        print(header)
        for wname in wdata:
            b = results[(wname, "base")][target_i]
            cells = []
            for m in mnames:
                c = results[(wname, m)][target_i]
                cells.append("%22s" % ("%.0f" % c if m == "base" else "%.0f (%.2fx)" % (c, b / c if c == c and c > 0 else float("nan"))))
            print("%-14s" % wname + "".join(cells))
    print("\nResults written to %s" % out_csv)


if __name__ == "__main__":
    main()
