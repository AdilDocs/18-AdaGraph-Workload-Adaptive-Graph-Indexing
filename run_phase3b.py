"""Phase 3b: emerging topics, geometric refresh, and data updates.

Epochs of queries whose distribution changes, optionally with insertions and deletions.
Per epoch: (1) apply the epoch's data updates to every index, (2) evaluate every index on
test queries from this epoch's distribution, with exact ground truth on the live data,
(3) AdaGraph serves the epoch's queries and adapts, (4) the rebuild reference is rebuilt.

Scenarios
    emerging  topics arrive in blocks; the initial history covers only the first block, and
              each phase brings topics never seen before (old topics keep 10% of traffic)
    topic     hot topics rotate, all topics present from the start (as in Phase 3a)
    modality  queries shift from images (in-distribution) to text (OOD)

Methods
    Vamana        query-agnostic, FreshDiskANN-style inserts/deletes
    Static-QA     query-aware graph built once; follows data updates but learns no new queries
    Rebuild-QA    query-aware graph rebuilt from scratch every epoch (quality reference)
    AdaGraph-noGR incremental query layer, geometric layer never refreshed (Phase 3a)
    AdaGraph      incremental query layer + budgeted geometric refresh

Examples:
    python run_phase3b.py --dataset t2i --n 200000 --scenario emerging
    python run_phase3b.py --dataset t2i --n 200000 --scenario modality --update-rate 0.01
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import numba

from adagraph.data import load_dataset, compute_groundtruth, recall_at_k
from adagraph.graph import VamanaGraph, batch_search, _pow2_at_least
from adagraph.dynamic import QueryAwareGraph, QueryWindow, DynamicVamana
from adagraph.workload import kmeans_centroids, _assign
from run_phase1 import interp_cost

L_EVAL = [10, 15, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 720, 1000]
METHODS = ["Vamana", "Static-QA", "Rebuild-QA", "AdaGraph"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "t2i"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n", type=int, default=200000, help="initial number of live vectors")
    ap.add_argument("--scenario", default="emerging", choices=["emerging", "topic", "modality"])
    ap.add_argument("--update-rate", type=float, default=0.0,
                    help="per epoch, insert and delete this fraction of n (0 = static data)")
    ap.add_argument("--geo", default="hybrid", choices=["hybrid", "enhance", "vamana"],
                    help="geometric layer: hybrid (enhancement searched on query layer + Vamana scaffold), "
                         "enhance (RoarGraph-style), or a query-independent Vamana layer")
    ap.add_argument("--scaffold-R", type=int, default=16, help="degree of the Vamana scaffold / layer")
    ap.add_argument("--geo-budget", type=float, default=0.0,
                    help="fraction of live nodes whose geometric edges are re-enhanced per epoch (0 = never; best for hybrid)")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--Nq", type=int, default=100)
    ap.add_argument("--L-serve", type=int, default=100)
    ap.add_argument("--epoch-size", type=int, default=25000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--phases", type=int, default=5)
    ap.add_argument("--window", type=int, default=400000)
    ap.add_argument("--topics", type=int, default=32)
    ap.add_argument("--hot-topics", type=int, default=6)
    ap.add_argument("--test-per-epoch", type=int, default=2000)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Scenario:
    def __init__(self, args, pools):
        self.a = args
        self.rng = np.random.default_rng(args.seed)
        self.pools = pools
        self.kind = args.scenario
        if self.kind in ("topic", "emerging"):
            c = kmeans_centroids(pools["hist"], args.topics, seed=args.seed)
            hl = _assign(pools["hist"], c)
            tl = _assign(pools["ood"], c)
            ok = [t for t in range(args.topics) if (hl == t).any() and (tl == t).any()]
            self.hist_members = {t: np.flatnonzero(hl == t) for t in ok}
            self.test_members = {t: np.flatnonzero(tl == t) for t in ok}
            self.order = list(self.rng.permutation(ok))

    def phase(self, e):
        return min(self.a.phases - 1, e * self.a.phases // (self.a.epochs + 1))

    def label(self, e):
        return ("%d" % self.phase(e)) if self.kind != "modality" else ("%.2f" % self.ood_share(e))

    def _block(self, p):
        h = self.a.hot_topics
        return [self.order[(p * h + j) % len(self.order)] for j in range(h)]

    def _topic_probs(self, e):
        p = self.phase(e)
        probs = {}
        if self.kind == "emerging":
            for t in self._block(p):
                probs[t] = probs.get(t, 0.0) + (0.9 if p > 0 else 1.0) / self.a.hot_topics
            seen = [t for q in range(p) for t in self._block(q)]
            for t in seen:
                probs[t] = probs.get(t, 0.0) + 0.1 / len(seen)
        else:
            hot = self._block(p)
            cold = [t for t in self.order if t not in hot]
            for t in hot:
                probs[t] = 0.9 / len(hot)
            for t in cold:
                probs[t] = 0.1 / max(1, len(cold))
        ts = list(probs.keys())
        pr = np.array([probs[t] for t in ts])
        return ts, pr / pr.sum()

    def _sample(self, members, e, m):
        ts, pr = self._topic_probs(e)
        counts = self.rng.multinomial(m, pr)
        return np.concatenate([self.rng.choice(members[t], size=c) for t, c in zip(ts, counts) if c > 0])

    def ood_share(self, e):
        e0, e1 = self.a.epochs // 5, 2 * self.a.epochs // 3
        return float(np.clip((e - e0) / max(1, e1 - e0), 0.0, 1.0))

    def history(self, e, m):
        if self.kind != "modality":
            return self.pools["hist"][self._sample(self.hist_members, e, m)]
        n_ood = int(round(m * self.ood_share(e)))
        a = self.pools["hist"][self.rng.integers(0, self.pools["hist"].shape[0], n_ood)]
        b = self.pools["idhist"][self.rng.integers(0, self.pools["idhist"].shape[0], m - n_ood)]
        q = np.vstack([a, b])
        return q[self.rng.permutation(q.shape[0])]

    def test(self, e, m):
        if self.kind != "modality":
            return self.pools["ood"][self._sample(self.test_members, e, m)]
        n_ood = int(round(m * self.ood_share(e)))
        a = self.pools["ood"][self.rng.integers(0, self.pools["ood"].shape[0], n_ood)]
        b = self.pools["id"][self.rng.integers(0, self.pools["id"].shape[0], m - n_ood)]
        return np.vstack([a, b])


def eval_cost(search_fn, queries, gt, k, live):
    recs, costs, bad = [], [], 0
    for L in L_EVAL:
        if L < k:
            continue
        ids, nd, _ = search_fn(queries, L, k)
        v = ids[ids >= 0]
        bad += int((~live[v]).sum())
        recs.append(recall_at_k(ids, gt, k))
        costs.append(float(nd.mean()))
    return interp_cost(recs, costs, 0.90), interp_cost(recs, costs, 0.95), bad


def main():
    args = parse_args()
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    use = [m.strip() for m in args.methods.split(",") if m.strip()]
    n0 = args.n if args.dataset != "synthetic" else min(args.n, 20000)
    ins_per_epoch = int(round(args.update_rate * n0))
    n_cap = n0 + args.epochs * ins_per_epoch
    tag = "%s_n%d" % (args.dataset, n0)
    cache = os.path.join(args.data_dir, "cache", tag)
    os.makedirs(cache, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    print("== AdaGraph Phase 3b | dataset=%s n0=%d scenario=%s updates=%.3f/epoch geo-budget=%.2f threads=%d"
          % (args.dataset, n0, args.scenario, args.update_rate, args.geo_budget, numba.get_num_threads()))

    n_idhist = 300000 if args.scenario == "modality" else 0
    base, pools = load_dataset(args.dataset, args.data_dir, n=n_cap, n_hist=1000000, n_idhist=n_idhist)
    base = np.ascontiguousarray(base, dtype=np.float32)
    if args.scenario == "modality" and ("idhist" not in pools or "id" not in pools):
        raise SystemExit("modality scenario needs held-out image vectors (use a smaller n)")
    live0 = np.zeros(n_cap, np.bool_)
    live0[:n0] = True
    print("[data] capacity %d vectors (%d live at start, %d inserted per epoch), pools: %s"
          % (n_cap, n0, ins_per_epoch, ", ".join("%s=%d" % (k, v.shape[0]) for k, v in pools.items())))

    # Vamana on the initial live vectors
    gname = "vamana_R32_L100_a1.20_p2.npz" if n_cap == n0 else "vamana_R32_prefix%d_cap%d.npz" % (n0, n_cap)
    gpath = os.path.join(cache, gname)
    if os.path.exists(gpath):
        vg = VamanaGraph.load(gpath, base[:n0])
    else:
        print("[graph] building Vamana R=32 on the initial %d vectors ..." % n0)
        vg = VamanaGraph(base[:n0], R=32, L_build=100, alpha=1.2, passes=2).build(verbose=False)
        vg.save(gpath)
    vam = DynamicVamana(vg, base, live0)

    geo_graph = None
    if args.geo in ("vamana", "hybrid"):
        R2 = args.M if args.geo == "vamana" else args.scaffold_R
        gname2 = ("vamana_R%d_L100_a1.20_p2.npz" % R2) if n_cap == n0 else \
            ("vamana_R%d_prefix%d_cap%d.npz" % (R2, n0, n_cap))
        gpath2 = os.path.join(cache, gname2)
        if os.path.exists(gpath2):
            vg2 = VamanaGraph.load(gpath2, base[:n0])
        else:
            print("[graph] building Vamana R=%d (%s) ..." % (R2, "layer" if args.geo == "vamana" else "scaffold"))
            vg2 = VamanaGraph(base[:n0], R=R2, L_build=100, alpha=1.2, passes=2).build(verbose=False)
            vg2.save(gpath2)
        geo_graph = DynamicVamana(vg2, base, live0)

    sc = Scenario(args, pools)

    # initial window from the epoch-0 distribution (one-time search on Vamana)
    t0 = time.time()
    q0 = np.ascontiguousarray(sc.history(0, args.window), dtype=np.float32)
    nb0, _, _ = vam.search(q0, 200, args.Nq)
    win_gr = QueryWindow(args.window, args.Nq)
    win_gr.add(nb0, 0)
    ada_gr = QueryAwareGraph(base, M=args.M, live=live0, geo=args.geo,
                             geo_graph=geo_graph).build(win_gr.nbrs, win_gr.times)
    print("[init] window %d queries, query-aware graph built in %.1fs" % (args.window, time.time() - t0))
    idx = {"AdaGraph": (ada_gr, win_gr, args.geo_budget)}
    if "AdaGraph-noGR" in use:
        idx["AdaGraph-noGR"] = (ada_gr.clone(), win_gr.clone(), 0.0)
    if "Static-QA" in use:
        idx["Static-QA"] = (ada_gr.clone(), win_gr.clone(), 0.0)
    rebuild = ada_gr.clone() if "Rebuild-QA" in use else None

    search = {"Vamana": vam.search}
    for m in ("Static-QA", "Rebuild-QA", "AdaGraph-noGR", "AdaGraph"):
        if m in use:
            search[m] = None  # resolved per epoch (rebuild object changes)
    maint = {m: 0.0 for m in search}
    rng = np.random.default_rng(args.seed + 99)
    live = live0.copy()
    next_id = n0

    rows_out = []
    print("\ncost@0.95 per epoch (distance computations, lower is better)")
    print("epoch phase " + "".join("%14s" % m for m in search) + "   AdaGraph maint")
    for e in range(1, args.epochs + 1):
        # 1. data updates
        if ins_per_epoch > 0:
            del_ids = rng.choice(np.flatnonzero(live), size=ins_per_epoch, replace=False).astype(np.int32)
            ins_ids = np.arange(next_id, next_id + ins_per_epoch, dtype=np.int32)
            next_id += ins_per_epoch
            t1 = time.time()
            vam.insert(ins_ids)
            vam.delete(del_ids)
            maint["Vamana"] += time.time() - t1
            for m, (g, w, gb) in idx.items():
                t1 = time.time()
                g.insert(ins_ids)
                dead = g.delete(del_ids)
                ch = w.clean(dead)
                if m == "Static-QA":
                    g.update(w.nbrs, w.times, ch, geo_budget=0.0)
                else:
                    g._pending = ch
                maint[m] += time.time() - t1
            live[ins_ids] = True
            live[del_ids] = False

        # 1b. rebuild reference: from scratch on the current live data and AdaGraph's window
        #     (same information AdaGraph has at evaluation time)
        if rebuild is not None and e > 1:
            t1 = time.time()
            gg = idx["AdaGraph"][0].geo_graph
            rebuild = QueryAwareGraph(base, M=args.M, live=live, geo=args.geo,
                                      geo_graph=gg.clone() if gg is not None else None).build(win_gr.nbrs, win_gr.times)
            maint["Rebuild-QA"] += time.time() - t1
        elif rebuild is not None and ins_per_epoch > 0:
            rebuild = idx["AdaGraph"][0].clone()

        # 2. evaluate on this epoch's distribution, exact ground truth on the live data
        tq = np.ascontiguousarray(sc.test(e, args.test_per_epoch), dtype=np.float32)
        lid = np.flatnonzero(live)
        gt = lid[compute_groundtruth(base[lid], tq, k=args.k, verbose=False)]
        res = {}
        for m in search:
            if m == "Vamana":
                fn = vam.search
            elif m == "Rebuild-QA":
                fn = rebuild.search
            else:
                fn = idx[m][0].search
            res[m] = eval_cost(fn, tq, gt, args.k, live)

        # 3. AdaGraph variants serve the epoch's queries and adapt
        q = np.ascontiguousarray(sc.history(e, args.epoch_size), dtype=np.float32)
        info = ""
        for m in ("AdaGraph-noGR", "AdaGraph"):
            if m not in idx:
                continue
            g, w, gb = idx[m]
            ids, nd, _ = g.search(q, args.L_serve, args.Nq)
            t1 = time.time()
            ch = w.add(ids, e)
            pend = getattr(g, "_pending", None)
            if pend is not None:
                ch = np.concatenate([ch, pend])
                g._pending = None
            g.update(w.nbrs, w.times, ch, geo_budget=gb)
            maint[m] += time.time() - t1
            if m == "AdaGraph":
                info = "%6.1fs (refreshed %d)" % (time.time() - t1, g.timing.get("refreshed_nodes", 0))

        print("%5d %5s " % (e, sc.label(e)) + "".join("%14.0f" % res[m][1] for m in search) + "   " + info)
        for m in search:
            if res[m][2] > 0:
                print("   WARNING: %s returned %d deleted/not-inserted ids" % (m, res[m][2]))
            rows_out.append([args.dataset, n0, args.scenario, args.update_rate, e, sc.label(e), m,
                             res[m][0], res[m][1], maint[m], int(live.sum())])

    out_csv = os.path.join(args.out_dir, "phase3b_%s_%s_u%g.csv" % (tag, args.scenario, args.update_rate))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n0", "scenario", "update_rate", "epoch", "phase", "method",
                    "cost@0.90", "cost@0.95", "cumulative_maintenance_s", "live"])
        w.writerows(rows_out)

    by = {m: np.array([[r[7], r[8]] for r in rows_out if r[6] == m]) for m in search}
    print("\n== Summary over %d epochs (fails = epochs where recall 0.95 was not reached even at L=%d)"
          % (args.epochs, L_EVAL[-1]))
    for ti, target in enumerate(("0.90", "0.95")):
        common = np.all(np.stack([np.isfinite(by[m][:, ti]) for m in search]), axis=0)
        vb = by["Vamana"][common, ti].mean() if common.any() else float("nan")
        print("\nrecall %s: mean cost over the %d epochs where every method reached it" % (target, int(common.sum())))
        print("%-14s %12s %18s %8s %18s" % ("method", "mean cost", "speedup vs Vamana", "fails", "total maintenance"))
        for m in search:
            c = by[m][common, ti].mean() if common.any() else float("nan")
            fails = int((~np.isfinite(by[m][:, ti])).sum())
            print("%-14s %12.0f %18.2f %8d %17.1fs" % (m, c, vb / c if c == c and c > 0 else float("nan"),
                                                       fails, maint[m]))
    print("\nResults written to %s" % out_csv)


if __name__ == "__main__":
    main()
