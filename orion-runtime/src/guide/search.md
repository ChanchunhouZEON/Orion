<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

Search with packed logical rows; internal padding is not part of the input.
The library returns IDs and counts, not recall or synthesized distances.

```rust
use orion_ann::{BaseInput, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions, Result};
use orion_ann::GraphSourcePolicy;
# let dir = std::env::temp_dir().join(format!("orion-doc-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
let values: Vec<f32> = (0..24).map(|x| x as f32 / 24.0).collect();
let options = IndexOptions {
    threads: 1,
    cache_dir: dir.join("build"),
    graph: GraphConfig { source: GraphSourcePolicy::Rust, degree: 4,
        build_l: 8, max_extra: 2, ..Default::default() },
    ..Default::default()
};
let mut index = Index::new(BaseInput::from_f32(&values, 3)?, options)?;
let queries = values[..6].to_vec(); // Two rows, dimension three.
index.calibrate(queries.clone(), 2, Some(8))?;
let result = index.search(QueryBatch::from_f32(&queries, 3)?, &SearchOptions::new(2, 8))?;
assert_eq!(result.ids.len(), 4);
for (ids, &count) in result.ids.chunks(result.k).zip(&result.counts) {
    for &id in &ids[..count as usize] { assert!((id as usize) < index.len()); }
    assert!(ids[count as usize..].iter().all(|&id| id == u32::MAX));
}
// A corrected input can be retried after an ordinary validation error.
assert!(QueryBatch::from_f32(&[0.0; 2], 3).is_err());
assert_eq!(index.search(QueryBatch::from_f32(&values[..3], 3)?, &SearchOptions::new(2, 8))?.counts.len(), 1);
# drop(index);
# std::fs::remove_dir_all(&dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Serve cosine queries in batches

For embeddings, select `SearchMetric::Cosine` when building the index. The
runtime normalizes the base and queries. Calibrate each requested `k` on
representative rows. That calibration can be reused across query batches and
search beam widths. The last batch may contain fewer queries.

```rust
use orion_ann::{BaseInput, ErrorCode, GraphConfig, Index, IndexOptions, QueryBatch, SearchOptions, SearchMetric};
# let dir = std::env::temp_dir().join(format!("orion-cosine-guide-{}", std::process::id()));
let embeddings: Vec<f32> = (1..=32).flat_map(|i| [i as f32, 1.0, 2.0]).collect();
let options = IndexOptions {
    metric: SearchMetric::Cosine,
    graph: GraphConfig { degree: 4, build_l: 8, max_extra: 2, ..Default::default() },
    ..Default::default()
};
# let options = IndexOptions { cache_dir: dir.clone(), threads: 1, ..options };
let mut index = Index::new(BaseInput::from_f32(&embeddings, 3)?, options)?;
let warmup = &embeddings[..8 * 3];
index.calibrate(warmup.to_vec(), 1, Some(8))?;
let error = index.search(QueryBatch::from_f32(&[1.0, 2.0, 3.0], 3)?, &SearchOptions::new(3, 8)).unwrap_err();
assert_eq!(error.code(), ErrorCode::InvalidState);
index.calibrate(warmup.to_vec(), 3, Some(8))?;

let queries = &embeddings[8 * 3..13 * 3];
let mut neighbors = Vec::new();
for batch in queries.chunks(2 * index.dimension()) {
    let hits = index.search(QueryBatch::from_f32(batch, 3)?, &SearchOptions::new(3, 12))?;
    for (row, count) in hits.ids.chunks_exact(hits.k).zip(hits.counts) {
        neighbors.push(row[..count as usize].to_vec());
    }
}
assert_eq!(neighbors.len(), 5);
# std::fs::remove_dir_all(dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Calibration and errors

Adaptive search requires a successful calibration for the requested `k`; absence
returns `InvalidState`. Calibration requires at least one query and retains a
separate record for each `k`. Calibration also prepares the selected sidecars
under exclusive access, so concurrent first searches cannot block the Rayon pool
on lazy initialization. Opening a calibrated snapshot restores its saved records
and prepares any missing recipe sidecars before publishing the handle.
For non-adaptive comparisons, use `benchmark::search_non_adaptive` with the
`benchmark` feature or record a diagnostic trace with `trace_search` and
`adaptive = false`. Both use approximate graph traversal.

Require `1 <= k <= point_count` and `k <= beam <= u32::MAX/4`. A nonmultiple query
length returns `DimensionMismatch`. Coordinates must be finite; for native u8
indices the f32 transport values must be integral and in `[0, 255]`. Empty search
batches are allowed after parameter and calibration validation. Validating one
request does not invalidate the index when it fails.

## Ordering and concurrency

`ids` is a row-major matrix of width `k`; `counts[q]` is the valid prefix length.
Rows preserve query order. `u32::MAX` marks missing neighbors and is never a valid
ID. Equal-distance order is not promised as a cross-platform tie-break rule.
`SearchResult` owns its output, so no native borrowed buffers outlive a call.

Rust `Index::search` borrows `&self`. Share one calibrated index through `Arc`
(or scoped thread borrows); every request supplies its own `SearchOptions`.
Concurrent batches use the same bounded Rayon pool and independently checked-out
scratch. Calibration still requires `&mut Index`, preventing overlap with search.
Python searches take a shared read lock and release the GIL; calibration takes
an exclusive write lock. NumPy input is copied while holding the GIL before any
worker reads it. Do not mutate an array concurrently with that initial copy.

`QueryBatch` borrows packed `f32` or `u8` coordinates, validates shape, and never
changes the caller's buffer. L2/IP f32 input is borrowed until kernel preparation;
native-u8 input stays bytes. Cosine normalization and representation conversion
use internal buffers. SIMD padding and per-query working memory still allocate
as required; borrowing the input does not promise allocation-free search.

`SearchResult` is defined in `orion-ann-core` and re-exported by `orion-ann`.
The batch core methods return the same type. `rows()` iterates valid IDs without
padding. Requesting scores is reserved for a later API addition; no extra exact
scoring pass is performed in this interface.
