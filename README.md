# StagedDiskANN

A two-phase graph-based approximate nearest neighbor search algorithm that accelerates DiskANN (Vamana) by adaptively switching between **navigation** (full graph traversal) and **reranking** (localized candidate refinement) during search.

## Core Idea

Standard Vamana search expands all neighbors at every step, even after the search has converged to a local region. StagedDiskANN observes that:

1. **Early in search (navigation)**: long-range "remote" neighbors are essential for escaping local minima and reaching the correct region.
2. **After convergence (reranking)**: only nearby "local" neighbors and high-quality pruned candidates contribute to improving results.

By detecting convergence and switching to a reduced candidate set, StagedDiskANN skips unnecessary distance computations while preserving recall.

## Architecture

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

- **local**: Bidirectional graph neighbors (both u->v and v->u exist). These are "true" near neighbors confirmed by the graph structure.
- **remote**: Unidirectional neighbors (only u->v). These are long-range navigation shortcuts placed by Vamana's pruning.
- **extra**: Top-k closest pruned candidates from the Vamana build process. These are points that were spatially close but removed by alpha-occlusion pruning. Stored with distances for quality ranking.

### Bidirectional Edge Detection

The local/remote split is determined by **bidirectionality** of graph edges, not by distance thresholds or fixed fractions. This provides a semantically meaningful and data-adaptive boundary:

- In sparse graphs (alpha=1.2): bidir rate drops sharply from ~100% to ~1% across the neighbor list, creating a clear split.
- In dense graphs (alpha=2.0): bidir rate declines more gradually, automatically adjusting the local zone size.

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
3. Take top-k closest as extra candidates.

This eliminates the need for the `key_neighbor_count` parameter and anchor-based cross-lookups from the original design.

### Build-Time Distance Maintenance

Under the `staged_diskann` feature, `VertexAndNeighbors` maintains a parallel `neighbor_dists: Vec<f32>` alongside neighbor IDs. This enables:

- **Sorted insertion** in `inter_insert`: new reverse edges are inserted at the correct distance-sorted position via binary search, maintaining neighbor order throughout the build.
- **Skip P1 sort**: The post-build `sort_neighbors_by_distance_in_place` step (which recomputed ~N*degree distances) is eliminated.
- **Early dataset release**: The dataset is freed immediately after the build phase (before candidate extraction), ensuring dataset and candidate_sets never coexist in memory.

## Key Optimizations

### Memory Efficiency

| Optimization | Impact |
|---|---|
| Slab indexed by location (not anchor) | max_extra per node bounded by slab cap, not cross-query accumulation |
| `max_extra` parameter caps stride | PhasedGraph stride = `HEADER(4) + max_degree + max_extra`, cache-line aligned |
| Dataset freed before candidate extraction | Peak memory reduced by ~50 MB on 100K datasets |
| Single-pass PhasedGraph build with stack buffers | No intermediate `Vec<Vec<u32>>` allocation |
| `AlignedBoxWithSlice<u32>` slab | Cache-line aligned, zero-copy reads |

### Search Performance

| Optimization | Impact |
|---|---|
| Bidir-based local/remote split | Data-adaptive, no distance info needed at search time |
| Admission-based convergence | More accurate than distance-based, works across all dataset types |
| Reversible convergence | Prevents permanent recall loss from false convergence |
| Early exit after convergence | Stops search when consecutive steps produce no PQ admissions |
| Two-slice reranking (`local + extra`) | Smaller working set after convergence |
| Prefetch pipeline | Next node's graph slot and vector prefetched during current expansion |

### Early Exit

After convergence, the priority queue often contains many unvisited candidates that will not improve the result. The `EarlyExitChecker` tracks consecutive expansion steps with zero admissions during the converged phase. When this count exceeds the configured limit, the search terminates immediately.

This is especially effective for high-dimensional data (e.g., GIST-960) where distance computation dominates runtime — early exit directly reduces the number of expensive distance calls rather than just the number of neighbors per step.

## Configuration

StagedDiskANN uses a lower alpha than DiskANN to produce more diverse graph edges (more remote shortcuts for navigation, more pruned candidates for reranking):

```yaml
defaults:
  diskann:
    alpha: 2.0
    graph_degree: 32
    build_search_list_size: 48
  staged:
    alpha: 1.2              # more aggressive pruning = more remote shortcuts
    graph_degree: 32
    build_search_list_size: 48
    base_local_count: 0     # 0 = bidir-based adaptive
    max_extra: 4
    window_size: 5
    threshold: 0.15         # admission rate threshold for convergence

datasets:
  gist:
    dimension: 960
    staged:
      alpha: 1.5            # high-dim needs denser graph for connectivity
```

The `alpha` parameter controls the pruning aggressiveness and should be tuned per dataset:
- **Low-to-medium dim (32-128)**: `alpha=1.2` works well, producing a sparse graph with clear local/remote separation.
- **High dim (960+)**: `alpha=1.5` or higher to maintain graph connectivity.

## Benchmark Results

All benchmarks: 100K points, 8 threads, Recall@10 vs QPS.

### Same-Recall QPS Improvement

QPS interpolated at identical Recall@10 targets (100K points, 8 threads):

| Dataset | Dimension | Recall@10 | DiskANN QPS | StagedDiskANN QPS | Improvement |
|---|---|---|---|---|---|
| SIFT | 128 | 0.95 | 110,418 | 129,925 | **+18%** |
| SIFT | 128 | 0.98 | 76,378 | 93,953 | **+23%** |
| GloVe-25 | 32 | 0.95 | 147,984 | 230,993 | **+56%** |
| GloVe-25 | 32 | 0.98 | 95,540 | 145,219 | **+52%** |
| GloVe-100 | 100 | 0.85 | 36,473 | 37,163 | **+2%** |
| GloVe-100 | 100 | 0.90 | 21,062 | 21,891 | **+4%** |
| GIST | 960 | 0.85 | 12,262 | 13,132 | **+7%** |
| GIST | 960 | 0.90 | 9,082 | 9,705 | **+7%** |
| GIST | 960 | 0.95 | 5,848 | 6,242 | **+7%** |

### QPS vs Recall@10 Curves

![QPS vs Recall](visualizations/qps_recall_all.png)

StagedDiskANN (orange) consistently achieves higher QPS than DiskANN (blue) at the same recall level across all four datasets. The advantage is most pronounced on lower-dimensional datasets where distance computation cost is smaller relative to the graph traversal overhead saved by skipping remote neighbors.

### Memory Comparison

Both algorithms use R=32. StagedDiskANN uses a lower alpha (sparser graph), resulting in comparable or lower memory:

| Dataset | DiskANN (R=32, alpha=2.0) | StagedDiskANN (R=32) | PhasedGraph | Delta |
|---|---|---|---|---|
| SIFT (128-dim) | 213 MB | 197 MB | 18.3 MB | **-8%** |
| GloVe-100 (100-dim) | 177 MB | 161 MB | 18.3 MB | **-9%** |

The PhasedGraph structure is only **18.3 MB** for 100K nodes (stride=48 u32, cache-line aligned). Total index memory is often lower because the sparser graph (lower alpha) stores fewer edges per node.

## Code Structure

```
staged_diskann/
  src/
    model/
      phased_graph.rs       # PhasedGraph: AlignedBoxWithSlice slab + bidir split + write queue
    algorithm/
      search/
        convergence.rs      # Admission-based convergence detector
        early_exit.rs       # Early exit after consecutive zero-admission steps
        in_mem_search.rs     # Two-phase greedy beam search with early exit
    index/
      builder.rs            # build_diskann_index / build_diskann_index_ex
      compressed_index.rs   # StagedDiskANN: PhasedGraph construction + search entry points

diskann/
  src/
    model/graph/
      vertex_and_neighbors.rs   # Distance-sorted neighbor maintenance (add_sorted, neighbor_dists)
    algorithm/prune/
      prune.rs                  # Slab writes: (distance, pruned_id) per location
    index/inmem_index/
      inmem_index.rs            # extract_graph_and_candidates: slab extraction + dataset early free
```
