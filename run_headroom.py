"""Headroom test: how much can a query-aware graph gain on this data at all?

Compares, in distance computations to reach recall 0.90 / 0.95:
    Vamana R32                 our base graph (Phases 1-2c)
    Vamana+star+e32 (best 2c)  best "add edges on top" strategy, 100K approximate history
    RoarGraph exact M=35       paper setting with exact neighbors of the training queries
    RoarGraph exact M=16       same, smaller degree (closer to Vamana memory)
    RoarGraph approx M=35      same training queries, approximate (ground-truth-free) neighbors
    RoarGraph approx-all M=35  approximate neighbors of all available history queries

Exact neighbors of the training queries are computed once and cached.

Examples:
    python run_headroom.py --dataset t2i --n 200000
    python run_headroom.py --dataset t2i --n 1000000 --n-train 100000
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import numba

from adagraph.data import load_dataset, cached_groundtruth
from adagraph.graph import VamanaGraph, batch_search, _pow2_at_least
from adagraph.adaptive import AdaptiveLayerV2
from adagraph.roargraph import RoarGraph
from adagraph.workload import QueryStreams
from run_phase1 import L_SWEEP, interp_cost
from run_phase2 import sweep


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "t2i"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--n-train", type=int, default=0,
                    help="training queries with exact neighbors (0 = same as n, max 200000 by default)")
    ap.add_argument("--n-hist", type=int, default=1000000, help="all history queries (approximate)")
    ap.add_argument("--Nq", type=int, default=100, help="neighbors per training query")
    ap.add_argument("--Lc", type=int, default=500, help="projection candidate set size")
    ap.add_argument("--L-ce", type=int, default=100, help="beam width for connectivity enhancement")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--stream-len", type=int, default=10000)
    ap.add_argument("--M-exact", default="35,16", help="degrees for RoarGraph on exact neighbors ('none' = skip)")
    ap.add_argument("--M-approx", default="35", help="degrees for RoarGraph on approximate neighbors of the training queries")
    ap.add_argument("--M-all", default="35", help="degrees for RoarGraph on approximate neighbors of all history")
    ap.add_argument("--skip", default="", help="comma list of methods to skip (e.g. star)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    return ap.parse_args()


def approx_neighbors(g, queries, K, L=200, block=100000):
    out = np.empty((queries.shape[0], K), np.int32)
    hcap = _pow2_at_least(8 * L * g.adj.shape[1])
    for s in range(0, queries.shape[0], block):
        q = np.ascontiguousarray(queries[s:s + block], dtype=np.float32)
        r, _, _ = batch_search(g.data, g.adj, g.deg, g.entry, q, L, K, hcap)
        out[s:s + block] = r
    return out


def main():
    args = parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    n_arg = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    tag = "%s_n%d" % (args.dataset, n_arg)
    cache = os.path.join(args.data_dir, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())
    print("== AdaGraph headroom test | dataset=%s n=%d threads=%d" % (args.dataset, n_arg, numba.get_num_threads()))

    base, pools = load_dataset(args.dataset, args.data_dir, n=n_arg, n_hist=args.n_hist)
    hist = pools["hist"]
    n_train = args.n_train if args.n_train > 0 else min(n_arg, 200000)
    n_train = min(n_train, hist.shape[0])
    train = hist[:n_train]
    print("[data] base %s, training queries %d (exact), history %d (approximate)" % (base.shape, n_train, hist.shape[0]))
    eval_pools = [p for p in ("id", "ood") if p in pools]
    gts = {p: cached_groundtruth(cache, p, base, pools[p], k=100) for p in eval_pools}

    # base Vamana graph
    gpath = os.path.join(cache, "vamana_R32_L100_a1.20_p2.npz")
    if os.path.exists(gpath):
        g = VamanaGraph.load(gpath, base)
    else:
        print("[graph] building Vamana R=32 ...")
        g = VamanaGraph(base, R=32, L_build=100, alpha=1.2, passes=2).build()
        g.save(gpath)

    mlist = lambda s_: [] if s_.strip().lower() in ("", "none") else [int(x) for x in s_.split(",") if x.strip()]
    M_exact, M_approx, M_all = mlist(args.M_exact), mlist(args.M_approx), mlist(args.M_all)

    # training-query neighbors (both cached on disk)
    exact = None
    if M_exact:
        t0 = time.time()
        exact = cached_groundtruth(cache, "train%d" % n_train, base, train, k=args.Nq)
        print("[train] exact %d-NN of %d training queries (%.1fs)" % (args.Nq, n_train, time.time() - t0))
    apath = os.path.join(cache, "approx_hist%d_K%d.npy" % (hist.shape[0], args.Nq))
    t0 = time.time()
    if os.path.exists(apath):
        approx_all = np.load(apath)
    else:
        approx_all = approx_neighbors(g, hist, args.Nq)
        np.save(apath, approx_all)
    print("[train] approximate %d-NN of %d history queries on Vamana (%.1fs)" % (args.Nq, hist.shape[0], time.time() - t0))
    approx_train = approx_all[:n_train]
    if exact is not None:
        overlap = np.mean([len(np.intersect1d(a, b)) for a, b in zip(approx_train[:2000], exact[:2000])]) / args.Nq
        print("[train] approximate vs exact neighbor overlap: %.3f" % overlap)

    # methods
    methods, info = {}, {}
    methods["Vamana R32"] = lambda q, L, k: g.search(q, L, k)
    info["Vamana R32"] = "avg degree %.1f, built %.0fs" % (g.deg.mean(), g.build_time)
    if "star" not in skip:
        lay = AdaptiveLayerV2(g, mode="star", select="prune", A=16, K=32, L_obs=100, entries=32)
        lay.fit(hist[:100000], verbose=False)
        methods["Vamana+star+e32"] = lambda q, L, k: lay.search(q, L, k)
        info["Vamana+star+e32"] = "avg degree %.1f" % lay.deg_c.mean()
    builds = [("RoarGraph exact M=%d" % M, exact, M) for M in M_exact]
    builds += [("RoarGraph approx M=%d" % M, approx_train, M) for M in M_approx]
    if hist.shape[0] > n_train:
        builds += [("RoarGraph approx-all M=%d" % M, approx_all, M) for M in M_all]
    for name, nb, M in builds:
        if name in skip:
            continue
        print("[build] %s from %d queries" % (name, nb.shape[0]))
        rg = RoarGraph(base, M=M, Lc=args.Lc, L_ce=args.L_ce).build(nb)
        methods[name] = (lambda r: (lambda q, L, k: r.search(q, L, k)))(rg)
        info[name] = "avg degree %.1f, built %.0fs" % (rg.deg.mean(), rg.build_time)

    # workloads (same seeds as earlier phases)
    qs = QueryStreams({p: pools[p] for p in eval_pools}, n_clusters=64)
    workloads = {}
    for p in eval_pools:
        workloads["%s-uniform" % p] = qs.uniform(p, args.stream_len, seed=1)
    workloads["ood-skewed"] = qs.skewed("ood", args.stream_len, seed=2)
    pids, rows, _ = qs.drift("ood", args.stream_len, phases=5, seed=3)
    workloads["ood-drift"] = (pids, rows)
    if "id" in eval_pools:
        workloads["id-to-ood"] = qs.id_to_ood("id", "ood", args.stream_len, seed=4)
    else:
        print("[note] no held-out image vectors at this n, so ID workloads are skipped")

    results, rows_out = {}, []
    for wname, (pids, rws) in workloads.items():
        gt = np.empty((pids.shape[0], args.k), np.int32)
        for j, name in enumerate(qs.names):
            m = pids == j
            gt[m] = gts[name][rws[m], :args.k]
        queries = qs.gather(pids, rws)
        for mname, fn in methods.items():
            t0 = time.time()
            recs, costs = sweep(fn, queries, gt, args.k)
            results[(wname, mname)] = (interp_cost(recs, costs, 0.90), interp_cost(recs, costs, 0.95))
            for L, r, c in zip([l for l in L_SWEEP if l >= args.k], recs, costs):
                rows_out.append([args.dataset, n_arg, wname, mname, L, r, c])
            print("  %-11s %-26s cost@0.90=%6.0f cost@0.95=%6.0f (%.0fs)"
                  % (wname, mname, results[(wname, mname)][0], results[(wname, mname)][1], time.time() - t0))

    out_csv = os.path.join(args.out_dir, "headroom_%s.csv" % tag)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "workload", "method", "L", "recall@%d" % args.k, "dist_per_query"])
        w.writerows(rows_out)

    print("\n== Methods")
    for m, s in info.items():
        print("  %-26s %s" % (m, s))
    wn = list(workloads.keys())
    for ti, target in enumerate((0.90, 0.95)):
        print("\n== Speedup over 'Vamana R32' in distance computations at recall %.2f (>1 is better)" % target)
        print("%-26s" % "method" + "".join("%12s" % w for w in wn))
        for m in methods:
            cells = []
            for w in wn:
                b, c = results[(w, "Vamana R32")][ti], results[(w, m)][ti]
                cells.append("%12.2f" % (b / c) if c == c and c > 0 and b == b else "%12s" % "-")
            print("%-26s" % m + "".join(cells))
    print("\nResults written to %s" % out_csv)


if __name__ == "__main__":
    main()
