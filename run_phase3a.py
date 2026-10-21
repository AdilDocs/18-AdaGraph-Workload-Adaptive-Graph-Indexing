"""Phase 3a: query drift over time (static data).

A stream of epochs of queries whose distribution changes. Before each epoch, every method
is evaluated on test queries drawn from that epoch's distribution (distance computations
to reach recall 0.90 / 0.95); then AdaGraph serves the epoch's queries, keeps their
top-Nq results in a sliding window and updates itself incrementally.

Scenarios
    topic     text queries; the hot topics (k-means clusters of history queries) change
              every phase
    modality  queries start as images (in-distribution) and shift to text (OOD)

Methods
    Vamana        static, query-agnostic (R=32)
    Static-QA     query-aware graph built once from the initial window, never updated
    Rebuild-QA    query-aware graph rebuilt from scratch from the current window every
                  epoch (quality reference; uses the same served-query neighbors)
    AdaGraph      incremental re-projection of changed pivots after every epoch

Examples:
    python run_phase3a.py --dataset t2i --n 200000 --scenario topic
    python run_phase3a.py --dataset t2i --n 200000 --scenario modality
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import numba

from adagraph.data import load_dataset, cached_groundtruth, recall_at_k
from adagraph.graph import VamanaGraph, batch_search, _pow2_at_least
from adagraph.dynamic import QueryAwareGraph, QueryWindow
from adagraph.workload import kmeans_centroids, _assign
from run_phase1 import interp_cost

L_EVAL = [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 720, 1000]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "t2i"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--scenario", default="topic", choices=["topic", "modality"])
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--Nq", type=int, default=100, help="results kept per served query")
    ap.add_argument("--L-serve", type=int, default=100, help="beam width when serving stream queries")
    ap.add_argument("--epoch-size", type=int, default=25000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--phases", type=int, default=5)
    ap.add_argument("--window", type=int, default=400000)
    ap.add_argument("--topics", type=int, default=32)
    ap.add_argument("--hot-topics", type=int, default=6)
    ap.add_argument("--test-per-epoch", type=int, default=2000)
    ap.add_argument("--rebuild-every", type=int, default=1, help="0 = never rebuild")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Scenario:
    """Gives, for each epoch, history rows to serve and test rows to evaluate."""

    def __init__(self, args, pools):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.pools = pools
        self.kind = args.scenario
        if self.kind == "topic":
            hist = pools["hist"]
            c = kmeans_centroids(hist, args.topics, seed=args.seed)
            hl = _assign(hist, c)
            tl = _assign(pools["ood"], c)
            self.hist_members = [np.flatnonzero(hl == t) for t in range(args.topics)]
            self.test_members = [np.flatnonzero(tl == t) for t in range(args.topics)]
            self.order = self.rng.permutation(args.topics)

    def phase(self, e):
        return min(self.args.phases - 1, e * self.args.phases // max(1, self.args.epochs + 1))

    def _topic_probs(self, e):
        a = self.args
        p = np.full(a.topics, 0.1 / max(1, a.topics - a.hot_topics))
        start = (self.phase(e) * a.hot_topics) % a.topics
        hot = self.order[[(start + j) % a.topics for j in range(a.hot_topics)]]
        p[hot] = 0.9 / a.hot_topics
        # topics without test queries cannot be evaluated; drop them from both streams
        for t in range(a.topics):
            if self.test_members[t].size == 0 or self.hist_members[t].size == 0:
                p[t] = 0.0
        return p / p.sum()

    def ood_share(self, e):
        a = self.args
        e0, e1 = a.epochs // 5, 2 * a.epochs // 3
        return float(np.clip((e - e0) / max(1, e1 - e0), 0.0, 1.0))

    def _sample_topic(self, members, probs, m):
        counts = self.rng.multinomial(m, probs)
        rows = [self.rng.choice(members[t], size=c) for t, c in enumerate(counts) if c > 0]
        return np.concatenate(rows)

    def history(self, e, m):
        """Query vectors to serve in epoch e."""
        if self.kind == "topic":
            return self.pools["hist"][self._sample_topic(self.hist_members, self._topic_probs(e), m)]
        n_ood = int(round(m * self.ood_share(e)))
        a = self.pools["hist"][self.rng.integers(0, self.pools["hist"].shape[0], n_ood)]
        b = self.pools["idhist"][self.rng.integers(0, self.pools["idhist"].shape[0], m - n_ood)]
        q = np.vstack([a, b])
        return q[self.rng.permutation(q.shape[0])]

    def test(self, e, m):
        """(pool name, rows) pairs for evaluation in epoch e."""
        if self.kind == "topic":
            return [("ood", self._sample_topic(self.test_members, self._topic_probs(e), m))]
        n_ood = int(round(m * self.ood_share(e)))
        out = []
        if n_ood > 0:
            out.append(("ood", self.rng.integers(0, self.pools["ood"].shape[0], n_ood)))
        if m - n_ood > 0:
            out.append(("id", self.rng.integers(0, self.pools["id"].shape[0], m - n_ood)))
        return out


def eval_cost(search_fn, queries, gt, k):
    recs, costs = [], []
    for L in L_EVAL:
        if L < k:
            continue
        ids, nd, _ = search_fn(queries, L, k)
        recs.append(recall_at_k(ids, gt, k))
        costs.append(float(nd.mean()))
    return interp_cost(recs, costs, 0.90), interp_cost(recs, costs, 0.95)


def main():
    args = parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    n_arg = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    tag = "%s_n%d" % (args.dataset, n_arg)
    cache = os.path.join(args.data_dir, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print("== AdaGraph Phase 3a | dataset=%s n=%d scenario=%s threads=%d"
          % (args.dataset, n_arg, args.scenario, numba.get_num_threads()))

    n_idhist = 300000 if args.scenario == "modality" else 0
    base, pools = load_dataset(args.dataset, args.data_dir, n=n_arg, n_hist=1000000, n_idhist=n_idhist)
    if args.scenario == "modality" and ("idhist" not in pools or "id" not in pools):
        raise SystemExit("modality scenario needs held-out image vectors (use n < 1M)")
    gts = {p: cached_groundtruth(cache, p, base, pools[p], k=100) for p in ("id", "ood") if p in pools}
    print("[data] base %s, pools: %s" % (base.shape, ", ".join("%s=%d" % (k, v.shape[0]) for k, v in pools.items())))

    gpath = os.path.join(cache, "vamana_R32_L100_a1.20_p2.npz")
    if os.path.exists(gpath):
        vam = VamanaGraph.load(gpath, base)
    else:
        print("[graph] building Vamana R=32 ...")
        vam = VamanaGraph(base, R=32, L_build=100, alpha=1.2, passes=2).build()
        vam.save(gpath)

    sc = Scenario(args, pools)

    # initial window: queries of epoch 0, neighbors from a one-time search on Vamana
    t0 = time.time()
    init_q = sc.history(0, args.window)
    hcap = _pow2_at_least(8 * 200 * vam.R)
    init_nbrs, _, _ = batch_search(vam.data, vam.adj, vam.deg, vam.entry, init_q, 200, args.Nq, hcap)
    win = QueryWindow(args.window, args.Nq)
    win.add(init_nbrs, 0)
    ada = QueryAwareGraph(base, M=args.M).build(win.nbrs, win.times)
    static = ada.copy_frozen()
    print("[init] window of %d queries; initial query-aware graph built in %.1fs (projection %.1fs, enhancement %.1fs)"
          % (args.window, time.time() - t0, ada.timing["projection_s"], ada.timing["enhancement_s"]))
    rebuild = ada.copy_frozen() if args.rebuild_every > 0 else None

    methods = {
        "Vamana": lambda q, L, k: vam.search(q, L, k),
        "Static-QA": lambda q, L, k: static.search(q, L, k),
    }
    if rebuild is not None:
        methods["Rebuild-QA"] = lambda q, L, k: rebuild.search(q, L, k)
    methods["AdaGraph"] = lambda q, L, k: ada.search(q, L, k)
    maint = {m: 0.0 for m in methods}
    hcap_serve = _pow2_at_least(8 * args.L_serve * ada.adj.shape[1])

    rows_out = []
    header = "epoch phase " + "".join("%12s" % m for m in methods) + "   ada-maint  rebuild"
    print("\ncost@0.95 per epoch (distance computations, lower is better)")
    print(header)
    for e in range(1, args.epochs + 1):
        # 1. evaluate on this epoch's distribution, before adapting to it
        parts = sc.test(e, args.test_per_epoch)
        tq = np.vstack([pools[p][r] for p, r in parts])
        tg = np.vstack([gts[p][r, :args.k] for p, r in parts])
        res = {}
        for m, fn in methods.items():
            res[m] = eval_cost(fn, tq, tg, args.k)

        # 2. AdaGraph serves the epoch and updates incrementally
        q = sc.history(e, args.epoch_size)
        ids, nd, _ = batch_search(ada.data, ada.adj, ada.deg, ada.entry, np.ascontiguousarray(q, dtype=np.float32),
                                  args.L_serve, args.Nq, hcap_serve)
        changed = win.add(ids, e)
        ada.update(win.nbrs, win.times, changed)
        t_ada = ada.timing["total_s"]
        maint["AdaGraph"] += t_ada

        # 3. reference: full rebuild from the same window
        t_rb = float("nan")
        if rebuild is not None and e % args.rebuild_every == 0:
            t1 = time.time()
            rb = QueryAwareGraph(base, M=args.M).build(win.nbrs, win.times)
            t_rb = time.time() - t1
            rebuild.adj, rebuild.deg, rebuild.entry = rb.adj, rb.deg, rb.entry
            maint["Rebuild-QA"] += t_rb

        ph = sc.phase(e) if args.scenario == "topic" else sc.ood_share(e)
        print("%5d %5s " % (e, ("%d" % ph) if args.scenario == "topic" else ("%.2f" % ph))
              + "".join("%12.0f" % res[m][1] for m in methods) + "   %8.1fs %7.1fs" % (t_ada, t_rb))
        for m in methods:
            rows_out.append([args.dataset, n_arg, args.scenario, e, ph, m, res[m][0], res[m][1],
                             maint[m], float(nd.mean())])

    out_csv = os.path.join(args.out_dir, "phase3a_%s_%s.csv" % (tag, args.scenario))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "scenario", "epoch", "phase_or_ood_share", "method",
                    "cost@0.90", "cost@0.95", "cumulative_maintenance_s", "ada_serve_dist_per_query"])
        w.writerows(rows_out)

    print("\n== Summary over %d epochs" % args.epochs)
    print("%-12s %14s %14s %16s %18s" % ("method", "mean cost@0.90", "mean cost@0.95", "speedup vs Vamana", "total maintenance"))
    by = {m: np.array([[r[6], r[7]] for r in rows_out if r[5] == m]) for m in methods}
    vb = np.nanmean(by["Vamana"][:, 1])
    for m in methods:
        c90, c95 = np.nanmean(by[m][:, 0]), np.nanmean(by[m][:, 1])
        print("%-12s %14.0f %14.0f %16.2f %17.1fs" % (m, c90, c95, vb / c95, maint[m]))
    print("\nResults written to %s" % out_csv)


if __name__ == "__main__":
    main()
