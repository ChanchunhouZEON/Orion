Configure an embedded index with [`crate::IndexOptions`]. These options do not
read CLI presets or environment overrides. Validation happens when an index is
created/opened, not when a struct is constructed or deserialized.

```rust
use orion_ann_runtime::{GraphConfig, IndexOptions};
use orion_ann_runtime::config::GraphSourcePolicy;
let options = IndexOptions {
    threads: 8,
    graph: GraphConfig {
        source: GraphSourcePolicy::Rust,
        degree: 32, build_l: 100, alpha: 1.2, max_extra: 16,
        ..Default::default()
    },
    memory_budget_gib: Some(64.0),
    cache_dir: "cache/service".into(),
    ..Default::default()
};
assert_eq!(options.graph.degree, 32);
// JSON has the same field names as Python's configuration object.
let restored: IndexOptions = serde_json::from_str(&serde_json::to_string(&options)?)?;
assert_eq!(restored.threads, 8);
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Selection and validation

- `Auto` reuses a compatible PA cache/staged export, then builds in memory with
  ParlayANN. The default `parlay` feature supports f32 and native u8; disabling it
  retains the f32 Rust fallback. Explicit `Rust`/`Parlayann` choices remain strict.
  Native u8 never silently becomes an f32 build. ParlayANN currently uses two
  unquantized construction passes and records the recipe in cache provenance.
- Supplying `staged_file` selects PA; combining it with `Rust` is an error.
- Dimensions and row counts come from input. Row padding is internal and does
  not change the logical dimension exposed by the API.
- `storage=None` follows input dtype. Native u8 supports exact L2 with the
  `none / native-l2-u8 / none` cascade. Other metric/storage/cascade combinations
  are checked together. Explicit cascade strings use the CLI vocabulary.
- `threads` must be positive and at most the compiled worker limit; each index
  owns its pool. `window_size` is in `1..=64`. `degree > 0`, `build_l >= degree`,
  and finite `alpha >= 1` are required. `max_points=0` is invalid.
- A memory budget is a finite, positive GiB estimate checked before loading.
  It is not an OS allocation limit or a guarantee against process OOM.

## Errors and ownership

Invalid settings produce the shared structured error. `InvalidArgument` covers
invalid bounds; `UnsupportedConfiguration` covers unsupported execution recipes;
`ResourceExhausted` describes an exceeded estimated budget. Match `ErrorCode`,
not message text. Options are owned/cloned by the session; editing the caller's
copy cannot change an existing index. No unsafe block is needed for this API.
