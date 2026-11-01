# orion-ann

Adaptive graph search for approximate nearest neighbors.

Part of [Orion ANN](https://github.com/ChanchunhouZEON/Orion). Requires Rust 1.92 or newer. Licensed under MIT.

The runtime-dimensional `Orion<T>` supports native f32/u8 bases, padded SIMD
storage, adaptive cascades and graph cache I/O. For an owned service API with
file/array input, configuration, calibration and snapshots, use
[`orion-ann-runtime`](https://docs.rs/orion-ann-runtime).

```toml
[dependencies]
orion-ann = "0.1.0"
orion-ann-diskann = "0.1.0"
orion-ann-vector = "0.1.0"
```

```rust
use orion_ann::{DynamicSearchConfig, Orion, PhasedGraph};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_vector::{L2F32Distance, PaddedLayout};
let base = PaddedDataset::from_packed(&[1.0f32, 2., 3., 4., 5., 6.],
    PaddedLayout::for_kernel::<L2F32Distance>(3)?)?;
let graph = PhasedGraph::build_from_partitions(
    &[(vec![1], vec![], vec![]), (vec![0], vec![], vec![])], 1, 0);
let index: Orion = Orion::from_phased_graph(base, graph, 0, None, None);
let result = index.search_l2(&[1., 2., 3.], DynamicSearchConfig {
    k: 1, search_list_size: 2, ..Default::default()
})?;
assert_eq!(result, vec![0]);
# Ok::<(), Box<dyn std::error::Error>>(())
```

Fallible operations use `Error`, `ErrorCode`, `ErrorReport` and `Result`.
Inspect error codes/details rather than parsing messages. These errors preserve
backend causes and do not log automatically. The low-level graph/cascade modules
require their documented invariants; they are not an untrusted graph-file parser.

`StaticOrion` and the fixed dimension constants remain compatibility interfaces
for historical benchmarks. `NaiveOrion`, async beam search and PQ exports are
research/low-level interfaces, not the recommended service entry point. They
are not formally deprecated or silently removed in 0.1.0.
