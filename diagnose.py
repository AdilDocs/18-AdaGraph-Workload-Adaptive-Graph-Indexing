"""Where does search cost go? (diagnostic for the base graph)

For each query, at a fixed beam width L, records how many distance computations were spent
    approach : before the first true top-10 neighbor is reached
    refine   : after that, until the search ends
and how many of the true top-10 were reached at all. Compares ID and OOD queries.

    python diagnose.py --dataset t2i --n 200000
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import numba
from numba import njit, prange

from adagraph.data import load_dataset, cached_groundtruth
from adagraph.graph import VamanaGraph, _l2, _hs_insert, _pow2_at_least


@njit(fastmath=True)
def _search_diag(data, adj, deg, entry, q, L, hcap, gt):
    mask = hcap - 1
    table = np.full(hcap, -1, np.int32)
    cid = np.empty(L, np.int32)
    cd = np.empty(L, np.float32)
    cx = np.zeros(L, np.bool_)
    _hs_insert(table, mask, entry)
    cid[0] = entry
    cd[0] = _l2(q, data[entry])
    size = 1
    ndist = 1
    first_hit = -1
    hits = 0
    for t in range(gt.shape[0]):
        if gt[t] == entry:
            hits += 1
            first_hit = 1
    while True:
        i = 0
        while i < size and cx[i]:
            i += 1
        if i == size:
            break
        cx[i] = True
        v = cid[i]
        for j in range(deg[v]):
            u = adj[v, j]
            if not _hs_insert(table, mask, u):
                continue
            d = _l2(q, data[u])
            ndist += 1
            for t in range(gt.shape[0]):
                if gt[t] == u:
                    hits += 1
                    if first_hit < 0:
                        first_hit = ndist
            if size == L and d >= cd[L - 1]:
                continue
            if size < L:
                size += 1
            p = size - 1
            while p > 0 and cd[p - 1] > d:
                cid[p] = cid[p - 1]
                cd[p] = cd[p - 1]
                cx[p] = cx[p - 1]
                p -= 1
            cid[p] = u
            cd[p] = d
            cx[p] = False
    if first_hit < 0:
        first_hit = ndist
    return ndist, first_hit, hits


@njit(parallel=True)
def _diag_batch(data, adj, deg, entry, queries, L, hcap, gts):
    nq = queries.shape[0]
    out = np.zeros((nq, 3), np.int64)
    for i in prange(nq):
        a, b, c = _search_diag(data, adj, deg, entry, queries[i], L, hcap, gts[i])
        out[i, 0] = a
        out[i, 1] = b
        out[i, 2] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "sift", "t2i"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--Ls", default="40,100,240")
    args = ap.parse_args()
    n_arg = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    cache = os.path.join(args.data_dir, "cache", "%s_n%d" % (args.dataset, n_arg))
    base, pools = load_dataset(args.dataset, args.data_dir, n=n_arg)
    g = VamanaGraph.load(os.path.join(cache, "vamana_R32_L100_a1.20_p2.npz"), base)
    print("== Cost breakdown on the base graph | %s n=%d" % (args.dataset, n_arg))
    print("%-5s %5s %12s %12s %12s %10s" % ("pool", "L", "total dist", "approach", "refine", "top10 seen"))
    for p in ("id", "ood"):
        if p not in pools:
            continue
        gt = cached_groundtruth(cache, p, base, pools[p], k=100)[:, :10].astype(np.int32)
        q = np.ascontiguousarray(pools[p], dtype=np.float32)
        for L in [int(x) for x in args.Ls.split(",")]:
            r = _diag_batch(g.data, g.adj, g.deg, g.entry, q, L, _pow2_at_least(8 * L * g.R), gt)
            tot, app, hits = r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean()
            print("%-5s %5d %12.0f %8.0f (%2.0f%%) %8.0f (%2.0f%%) %10.2f"
                  % (p, L, tot, app, 100 * app / tot, tot - app, 100 * (tot - app) / tot, hits))


if __name__ == "__main__":
    main()
