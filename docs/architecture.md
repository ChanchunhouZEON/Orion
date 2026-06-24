# Architecture

[← Back to main README](../README.md)


### PhasedGraph

A cache-line aligned, concurrency-safe graph structure backed by `AlignedBoxWithSlice<u32>` with per-node reader tracking and bounded write queue (same concurrency model as DiskANN's `NodeSlabBuffer`).

Each node's neighbors are partitioned into three zones:

```
+------------------+-------------------+------------------+
| local_neighbors  | remote_neighbors  | extra_candidates |
| (navigation +    | (navigation only) | (reranking only) |
|  reranking)      |                   |                  |
+------------------+-------------------+------------------+
```

- **local**: **Top 60% of `(original_neighbours ∪ extras)` sorted ascending by distance.** Origin neighbours and extras are merged into one distance-sorted pool, and the closest `round(0.6 × |pool|)` entries form the local zone. This pulls in extras that are closer than the originals' tail (occlusion-pruned candidates spatially nearer than the surviving Vamana neighbours), while pushing the originals' distance tail out into remote.
- **remote**: The bottom 40% of the merged-and-sorted pool — long-range candidates that participate in pre-converged navigation but are skipped during reranking.
- **extra**: Top-k closest pruned candidates from the Vamana build process — points that were spatially close but removed by alpha-occlusion pruning. They participate in the merge above but are also kept as a separate zone for converged / rerank-phase consultation outside the local cut.

### Distance-Percentile Local/Remote Split

The local/remote split is a **distance-rank cut at the 60th percentile of `(original_neighbours ∪ extras)`**, sorted ascending by build-time distance:

- **Merge before cut.** Vamana's surviving neighbours and the per-node pruned-candidate `extras` are unioned into a single distance-sorted pool, then sliced at `round(0.6 × |pool|)`. Originals that are qualified as local get placed into the local zone; the counterpart originals get pushed out to remote; Extras that are closer than the originals' distance tail get placed into the candidate zone. This is what gives reranking a higher-quality candidate set than a strictly origin-only split would.
- **Per-node, length-relative.** The cut scales with each node's actual pool size, not the global `R` — variable-degree nodes (Vamana's prune produces uneven degrees) keep a consistent 60/40 split rather than a uniform absolute count.
- **No build-time graph crawl needed.** Distances come straight from Vamana's prune step (originals) and the per-node candidate slab (extras) — partitioning is a single sort + slice.

The cutoff is tunable via `ORION_LOCAL_PCT`; the default 60 is a empirical value for best searching configuration.

### Admission-Based Convergence Detection

Convergence is detected by tracking the **admission rate** of candidates into the priority queue, rather than distance-based heuristics:

- A sliding window tracks what fraction of recent expansion steps produced at least one PQ admission.
- When the admission fraction drops below a threshold (default 15%), the search switches to reranking mode.
- Convergence is **reversible**: if a burst of admissions occurs, the search re-enters navigation mode.
- A minimum visit count (2x window size) prevents premature convergence.

This directly measures "is the search still making progress?" and works consistently across datasets with different distance distributions.

### Distance-Sorted Candidate Slab

During Vamana build, the occlusion pruning process records `(distance, pruned_id)` pairs in a per-node slab (mmap-backed, indexed by location node). At extract time:

1. Read `slab[node]`: all `(distance, pruned_id)` pairs from pruning events involving this node.
2. Sort by distance, dedup by ID.
3. Take top-k{default 16} closest as extra candidates.

This eliminates the need for the `key_neighbor_count` parameter and anchor-based cross-lookups from the original design.

### Build-Time Distance Maintenance

Under the `diskann` crate's `orion` cargo feature, `VertexAndNeighbors` maintains a parallel `neighbor_dists: Vec<f32>` alongside neighbor IDs. This enables:

- **Sorted insertion** in `inter_insert`: new reverse edges are inserted at the correct distance-sorted position via binary search, maintaining neighbor order throughout the build.
- **Early dataset release**: The dataset is freed immediately after the build phase (before candidate extraction), ensuring dataset and candidate_sets never coexist in memory.
