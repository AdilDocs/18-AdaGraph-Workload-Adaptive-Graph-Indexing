"""Phase 2c: more history, and comparison at equal memory.

Starting from the best Phase 2b strategy (star edges + 32 history entry points), tests
    - history size: 100K vs 1M history queries
    - star size:    top-32 vs top-64 results per history query
    - equal total degree: base graph with fewer edges (R=24 or 16) plus adaptive edges,
      so that base + adaptive = 32 edges per node, the same as the R=32 baseline.

Examples:
    python run_phase2c.py --dataset t2i  --n 200000
    python run_phase2c.py --dataset sift --n 200000 --n-hist 800000
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

# name: (base R, star K, adaptive budget A, entry points, history size key)
CONFIGS = [
    ("R32 base",             32, 0,  0,  0,  None),
    ("R32+star32+e32 h100K", 32, 32, 16, 32, "small"),
    ("R32+star32+e32 h1M",   32, 32, 16, 32, "large"),
    ("R32+star64+e32 h1M",   32, 64, 16, 32, "large"),
    ("R24 base",             24, 0,  0,  0,  None),
    ("R24+star32/A8+e32",    24, 32, 8,  32, "large"),
    ("R16+star32/A16+e32",   16, 32, 16, 32, "large"),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "sift", "t2i"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--L-build", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--stream-len", type=int, default=10000)
    ap.add_argument("--L-obs", type=int, default=100)
    ap.add_argument("--n-hist", type=int, default=1000000, help="'large' history size")
    ap.add_argument("--n-hist-small", type=int, default=100000, help="'small' history size")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def get_graph(cache, base, R, args):
    path = os.path.join(cache, "vamana_R%d_L%d_a%.2f_p%d.npz" % (R, args.L_build, args.alpha, args.passes))
    if os.path.exists(path):
        return VamanaGraph.load(path, base)
    print("[graph] building base graph R=%d ..." % R)
    g = VamanaGraph(base, R=R, L_build=args.L_build, alpha=args.alpha, passes=args.passes).build()
    g.save(path)
    return g


def main():
    args = parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    n_arg = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    tag = "%s_n%d" % (args.dataset, n_arg)
    cache = os.path.join(args.data_dir, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print("== AdaGraph Phase 2c | dataset=%s n=%d threads=%d" % (args.dataset, n_arg, numba.get_num_threads()))

    base, pools = load_dataset(args.dataset, args.data_dir, n=n_arg, n_hist=args.n_hist)
    hist_all = pools["hist"]
    hist_sizes = {"small": min(args.n_hist_small, hist_all.shape[0]), "large": hist_all.shape[0]}
    print("[data] history pool: %d queries (small=%d, large=%d)" % (hist_all.shape[0], hist_sizes["small"], hist_sizes["large"]))
    eval_pools = [p for p in ("id", "ood") if p in pools]
    gts = {p: cached_groundtruth(cache, p, base, pools[p], k=100) for p in eval_pools}

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

    graphs, observed = {}, {}
    methods, info = {}, {}
    for name, R, K, A, ent, hkey in CONFIGS:
        if R not in graphs:
            graphs[R] = get_graph(cache, base, R, args)
        g = graphs[R]
        if K == 0:
            methods[name] = (lambda g_: (lambda q, L, k: g_.search(q, L, k)))(g)
            info[name] = "avg degree %.1f" % g.deg.mean()
            continue
        lay = AdaptiveLayerV2(g, mode="star", select="prune", A=A, K=K, L_obs=args.L_obs,
                              alpha=args.alpha, entries=ent)
        if R not in observed:
            # observe all history once per base graph with the largest K, reuse for every config
            kmax = max(c[2] for c in CONFIGS if c[1] == R)
            t0 = time.time()
            obs_layer = AdaptiveLayerV2(g, mode="star", K=kmax, L_obs=max(args.L_obs, kmax))
            observed[R] = obs_layer.observe(hist_all)
            print("[observe] R=%d: %d history queries, top-%d, %.1fs" % (R, hist_all.shape[0], kmax, time.time() - t0))
        nh = hist_sizes[hkey]
        res, exp = observed[R]
        print("[fit] %s" % name)
        lay.fit(hist_all[:nh], observed=(res[:nh], exp[:nh]))
        methods[name] = (lambda l_: (lambda q, L, k: l_.search(q, L, k)))(lay)
        info[name] = "avg degree %.1f (base %.1f + adaptive %.1f), %d entries, %d history" % (
            lay.deg_c.mean(), g.deg.mean(), lay.deg_c.mean() - g.deg.mean(), lay.stats["entries"], nh)

    results, rows_out = {}, []
    for wname, (queries, gt) in wdata.items():
        for mname, fn in methods.items():
            t0 = time.time()
            recs, costs = sweep(fn, queries, gt, args.k)
            results[(wname, mname)] = (interp_cost(recs, costs, 0.90), interp_cost(recs, costs, 0.95))
            for L, r, c in zip([l for l in L_SWEEP if l >= args.k], recs, costs):
                rows_out.append([args.dataset, n_arg, wname, mname, L, r, c])
            print("  %-12s %-22s cost@0.90=%6.0f cost@0.95=%6.0f (%.0fs)"
                  % (wname, mname, results[(wname, mname)][0], results[(wname, mname)][1], time.time() - t0))

    out_csv = os.path.join(args.out_dir, "phase2c_%s.csv" % tag)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "workload", "method", "L", "recall@%d" % args.k, "dist_per_query"])
        w.writerows(rows_out)

    print("\n== Methods")
    for m, s in info.items():
        print("  %-22s %s" % (m, s))
    wn = list(wdata.keys())
    ref = "R32 base"
    for ti, target in enumerate((0.90, 0.95)):
        print("\n== Speedup over '%s' in distance computations at recall %.2f (>1 is better)" % (ref, target))
        print("%-22s" % "method" + "".join("%13s" % w for w in wn))
        for m in methods:
            cells = []
            for w in wn:
                b, c = results[(w, ref)][ti], results[(w, m)][ti]
                cells.append("%13.2f" % (b / c) if c == c and c > 0 and b == b else "%13s" % "-")
            print("%-22s" % m + "".join(cells))
    print("\nResults written to %s" % out_csv)


if __name__ == "__main__":
    main()
