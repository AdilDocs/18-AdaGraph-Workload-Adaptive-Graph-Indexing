# AdaGraph — Phase 1 testbed

Static Vamana graph + beam search + workload generator. Phase 1 measures how many
distance computations a query needs to reach a recall target (the cost c(q;G) of Eq. (2)),
for in-distribution, out-of-distribution, skewed and drifting workloads.

## Setup (once)

Put this folder's contents next to your `data` folder, so the layout is:

```
18-Paper Working TKDE2/
    data/sift/sift-128-euclidean.hdf5
    data/t2i/t2i_base_1M.fbin, t2i_query_10K.fbin, t2i_learn_1M.fbin
    adagraph/            (package)
    run_phase1.py
```

Install the packages in your virtual environment:

```
pip install -r requirements.txt
```

## Run

```
python run_phase1.py --dataset synthetic          # 1-minute smoke test, no files needed
python run_phase1.py --dataset sift --n 200000
python run_phase1.py --dataset t2i  --n 200000
```

The first run compiles the numba code (about 20 s) and builds the graph; both the graph
and the ground truth are cached under `data/cache/`, so later runs start in seconds.
Results go to `results/phase1_<dataset>_n<n>.csv`, and a summary is printed at the end.

Useful options: `--n 1000000` (full 1M), `--R 32 --L-build 100 --alpha 1.2` (graph),
`--threads 8` (default: all cores), `--stream-len 10000`.

## What to send back

The final **Summary** block of each run (or the CSV files). The key number is how many
distance computations OOD queries need compared with ID queries at recall 0.90/0.95.

## Phase 2: adaptive edges (go/no-go test)

```
python run_phase2.py --dataset t2i  --n 200000
python run_phase2.py --dataset sift --n 200000
```

Reuses the cached graph and ground truth from Phase 1. Builds adaptive edges from the
search results of 100K history queries (no ground truth), for budgets A = 8 and 16 edges
per node, and prints the distance computations needed for recall 0.90/0.95 next to the
base graph, with the speedup (x). Options: `--A 8,16,32`, `--K-obs 10`, `--L-obs 100`,
`--min-count 2` (keep only edges supported by at least 2 history queries).

## Phase 2b: which adaptive strategy works?

```
python diagnose.py    --dataset t2i  --n 200000
python run_phase2b.py --dataset t2i  --n 200000
python run_phase2b.py --dataset sift --n 200000
```

`diagnose.py` splits the search cost of ID and OOD queries into *approach* (before the
first true neighbor is reached) and *refine* (after). `run_phase2b.py` compares eight
ground-truth-free strategies (entry points, clique / star / path edges, prune vs count
selection) and prints speedups over the base graph at recall 0.90 and 0.95.
Run a subset with e.g. `--variants base,star-prune,star-prune+e32`.

## Phase 2c: more history, equal memory

```
python run_phase2c.py --dataset t2i  --n 200000
python run_phase2c.py --dataset sift --n 200000 --n-hist 800000
```

Best Phase 2b strategy (star edges + 32 entry points) with 100K vs 1M history queries,
top-32 vs top-64 stars, and base graphs with fewer edges (R=24, R=16) plus adaptive
edges so the total degree stays at 32. Builds the R=24/R=16 graphs once (cached).
Needs about 4-5 GB of RAM for 1M history queries on T2I.

## Headroom test: query-aware graph built from scratch (RoarGraph-style)

```
python run_headroom.py --dataset t2i --n 200000                       # ~20 min
python run_headroom.py --dataset t2i --n 1000000 --n-train 100000     # ~1 hour, optional
```

Compares the Vamana baseline, the best Phase 2c layer, and a RoarGraph re-implementation
built from exact (upper bound) or approximate (ground-truth-free) neighbors of history
queries. Exact training neighbors are cached in `data/cache/`.

## Phase 3a: query drift over time

```
python run_phase3a.py --dataset t2i --n 200000 --scenario topic      # ~20 min
python run_phase3a.py --dataset t2i --n 200000 --scenario modality   # ~20 min
```

30 epochs of 25K queries. `topic`: the hot text-query topics change every 6 epochs.
`modality`: queries shift from images (in-distribution) to text (OOD). Before each epoch
every method is evaluated on that epoch's distribution; then AdaGraph serves the epoch,
keeps the top-100 results of each query in a 400K-query window, and re-projects only the
pivots that changed. Compared with Vamana, a query-aware graph built once (Static-QA) and
a full rebuild every epoch (Rebuild-QA).

## Phase 3b: emerging topics, geometric refresh, data updates

```
python run_phase3b.py --dataset t2i --n 200000 --scenario emerging                     # ~40 min
python run_phase3b.py --dataset t2i --n 200000 --scenario modality --update-rate 0.01  # ~45 min
```

`emerging`: the initial history covers one block of 6 topics; each phase brings 6 topics
never seen before. `--update-rate 0.01`: every epoch inserts and deletes 1% of the vectors
(Vamana gets FreshDiskANN-style updates). Ground truth is recomputed exactly on the live
vectors every epoch. Default `--geo hybrid` (final design): the geometric layer is
built by connectivity enhancement whose searches run on the query layer plus a
query-independent Vamana scaffold (R=16) that is used only for construction and
maintenance; it is updated only by insertions/deletions (`--geo-budget 0`). Alternatives
for ablations: `--geo enhance` (RoarGraph-style) and `--geo vamana`. The summary reports, per recall target, the mean cost over epochs where
every method reached the target, and how many epochs each method failed to reach it.

## Files

| File | Content |
|---|---|
| `adagraph/data.py` | SIFT/T2I loaders, inner-product to L2 conversion, cached exact ground truth |
| `adagraph/graph.py` | Vamana build (parallel batch insertion, RobustPrune), beam search with distance counting and search traces |
| `adagraph/workload.py` | Uniform, skewed (Zipf over clusters), drifting and ID-to-OOD query streams; insert/delete operation streams |
| `adagraph/adaptive.py` | Adaptive edge layer from co-occurring search results, pruned to budget A |
| `run_phase1.py` | Phase 1 experiment driver |
| `run_phase2.py` | Phase 2: base graph vs adaptive edges |
| `run_phase2b.py` | Phase 2b: comparison of eight adaptive strategies |
| `run_phase2c.py` | Phase 2c: history size, star size, equal-degree comparison |
| `adagraph/roargraph.py` | RoarGraph construction: projection, AcquireNeighbors, connectivity enhancement |
| `run_headroom.py` | Headroom test: Vamana vs RoarGraph (exact / approximate neighbors) |
| `adagraph/dynamic.py` | AdaGraph: query window, two-layer query-aware graph, incremental re-projection |
| `run_phase3b.py` | Phase 3b: emerging topics, geometric refresh, insertions/deletions |
| `run_phase3a.py` | Phase 3a: drifting query streams (topic shift, image-to-text shift) |
| `diagnose.py` | Cost breakdown (approach vs refine) for ID and OOD queries |
