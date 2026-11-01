Search with packed logical rows; internal padding is not part of the input.
The library returns IDs and counts, not recall or synthesized distances.

```rust
use orion_ann_runtime::{BaseInput, GraphConfig, Index, IndexOptions, Result};
use orion_ann_runtime::config::GraphSourcePolicy;
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
let result = index.search(queries, 2, 8, true)?;
assert_eq!(result.ids.len(), 4);
for (ids, &count) in result.ids.chunks(result.k).zip(&result.counts) {
    for &id in &ids[..count as usize] { assert!((id as usize) < index.len()); }
    assert!(ids[count as usize..].iter().all(|&id| id == u32::MAX));
}
// A corrected input can be retried after an ordinary validation error.
assert!(index.search(vec![0.0; 2], 2, 8, false).is_err());
assert_eq!(index.search(values[..3].to_vec(), 2, 8, false)?.counts.len(), 1);
# drop(index);
# std::fs::remove_dir_all(&dir)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Calibration and errors

Adaptive search requires a successful calibration for the requested `k`; absence
returns `InvalidState`. Calibration requires at least one query and retains a
separate record for each `k`. `adaptive=false` disables convergence and early-exit
thresholds, but the graph/cascade search remains approximate, not exhaustive.

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

Rust search/calibration require `&mut Index`. Serialize requests per handle or
use separate owned indices; internal batch parallelism uses its configured pool.
Python also serializes native operations on one handle and releases the GIL
around CPU work. It does not imply simultaneous searches on that same handle.
