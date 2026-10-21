"""Phase 2b: which adaptive strategy works on real data?

Compares, on the same workloads as Phases 1-2, the distance computations needed to reach
recall 0.90 / 0.95 for several ground-truth-free strategies built from history queries:

    base             static Vamana graph
    entry32          base graph + 32 history-based entry points
    clique-prune     Phase 2 default (pairs among top-10 results, RobustPrune), A=16
    star-prune       best result <-> top-32 results, RobustPrune, A=16
    star-count       best result <-> top-32 results, most-supported edges, A=16
    path-count       early search-path nodes -> final result, most-supported edges, A=16
    star-prune+e32   star-prune + 32 entry points
    path-count+e32   path-count + 32 entry points

Examples:
    python run_phase2b.py --dataset t2i  --n 200000
    python run_phase2b.py --dataset sift --n 200000
    python run_phase2b.py --dataset t2i  --variants base,entry32,star-prune+e32
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import numba

from adagraph.data import load_dataset, cached_groundtruth
from adagraph.graph import VamanaGraph
from adagraph.adaptive import AdaptiveLayerV2
from adagraph.workload import QueryStreams
from run_phase1 import L_SWEEP, interp_cost
from run_phase2 import sweep

VARIANTS = {
    # name:            (mode,     select,  K,  entries)
    "base":            (None,     None,    0,  0),
    "entry32":         ("none",   None,    0,  32),
    "clique-prune":    ("clique", "prune", 10, 0),
    "star-prune":      ("star",   "prune", 32, 0),
    "star-count":      ("star",   "count", 32, 0),
    "path-count":      ("path",   "count", 10, 0),
    "star-prune+e32":  ("star",   "prune", 32, 32),
    "path-count+e32":  ("path",   "count", 10, 32),
}


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
    ap.add_argument("--A", type=int, default=16)
    ap.add_argument("--L-obs", type=int, default=100)
    ap.add_argument("--n-hist", type=int, default=100000)
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    n_arg = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    tag = "%s_n%d" % (args.dataset, n_arg)
    cache = os.path.join(args.data_dir, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print("== AdaGraph Phase 2b | dataset=%s n=%d threads=%d" % (args.dataset, n_arg, numba.get_num_threads()))

    base, pools = load_dataset(args.dataset, args.data_dir, n=n_arg)
    hist = pools["hist"][:args.n_hist]
    eval_pools = [p for p in ("id", "ood") if p in pools]
    gts = {p: cached_groundtruth(cache, p, base, pools[p], k=100) for p in eval_pools}

    gpath = os.path.join(cache, "vamana_R%d_L%d_a%.2f_p%d.npz" % (args.R, args.L_build, args.alpha, args.passes))
    if os.path.exists(gpath):
        g = VamanaGraph.load(gpath, base)
    else:
        g = VamanaGraph(base, R=args.R, L_build=args.L_build, alpha=args.alpha, passes=args.passes).build()
        g.save(gpath)

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

    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    methods, info = {}, {}
    for name in names:
        mode, select, K, ent = VARIANTS[name]
        if mode is None:
            methods[name] = lambda q, L, k: g.search(q, L, k)
            info[name] = "%d base edges" % int(g.deg.sum())
            continue
        print("[fit] %s" % name)
        lay = AdaptiveLayerV2(g, mode=mode, select=select or "prune", A=args.A, K=max(K, 2),
                              L_obs=args.L_obs, alpha=args.alpha, entries=ent).fit(hist)
        methods[name] = (lambda lay_: (lambda q, L, k: lay_.search(q, L, k)))(lay)
        s = lay.stats
        info[name] = "+%d edges (+%.1f%%), %d entries, fit %.1fs" % (
            s["adaptive_edges"], s["extra_memory_pct"], s["entries"], s["fit_time_s"])

    g.search(base[:8], L=10, k=args.k)
    results, rows_out = {}, []
    for wname, (queries, gt) in wdata.items():
        for mname, fn in methods.items():
            t0 = time.time()
            recs, costs = sweep(fn, queries, gt, args.k)
            results[(wname, mname)] = (interp_cost(recs, costs, 0.90), interp_cost(recs, costs, 0.95))
            for L, r, c in zip([l for l in L_SWEEP if l >= args.k], recs, costs):
                rows_out.append([args.dataset, n_arg, wname, mname, L, r, c])
            print("  %-12s %-16s cost@0.90=%6.0f cost@0.95=%6.0f (%.0fs)"
                  % (wname, mname, results[(wname, mname)][0], results[(wname, mname)][1], time.time() - t0))

    out_csv = os.path.join(args.out_dir, "phase2b_%s.csv" % tag)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "workload", "method", "L", "recall@%d" % args.k, "dist_per_query"])
        w.writerows(rows_out)

    print("\n== Methods")
    for m, s in info.items():
        print("  %-16s %s" % (m, s))
    wn = list(wdata.keys())
    for ti, target in enumerate((0.90, 0.95)):
        print("\n== Speedup over base in distance computations at recall %.2f (>1 is better)" % target)
        print("%-16s" % "method" + "".join("%13s" % w for w in wn))
        for m in methods:
            cells = []
            for w in wn:
                b = results[(w, "base")][ti] if "base" in methods else float("nan")
                c = results[(w, m)][ti]
                cells.append("%13.2f" % (b / c) if c == c and c > 0 else "%13s" % "-")
            print("%-16s" % m + "".join(cells))
    print("\nResults written to %s" % out_csv)


if __name__ == "__main__":
    main()
