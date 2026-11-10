<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

Search with packed logical rows; internal padding is not part of the input.
The library returns IDs and counts, with optional exact candidate scores. It
does not calculate recall unless an application supplies its own ground truth.

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

Core `SearchResult` contains compact internal IDs, valid counts and optional
scores. Runtime `SearchResult` moves these buffers without copying and adds an
optional business-ID buffer. `rows()` iterates valid internal IDs without padding.
The default request performs no extra exact scoring pass. Equal-distance tie
ordering is not a cross-platform stability guarantee; use IDs to distinguish
neighbors and counts to handle fewer-than-k results.

## Optional RAM residency

Calibration prepares the search requirements. `pinning_hints()` reports each
prepared dataset's total payload; `pin()` only locks existing allocations and
never builds or reloads a sidecar. It pins RAM, not the CPU's L1/L2 caches.
The default policy attempts smaller datasets first with a 64 MiB page-rounded
budget. OS limits can still deny a request; inspect the report.

Ordinary builds use the measured prefetch schedule: 32-cache-line near windows
for both quantized scoring and reranking, a maximum 32-line far burst, and an
8-line compute-group trigger. These defaults also apply to Python and CLI
searches; benchmark-only environment overrides are not needed to enable them.
Prefetch scheduling and OS RAM pinning are independent mechanisms.

The CLI (including snapshot search) and `orion-sweep` automatically attempt to pin prepared
base/sidecar requirements after resource preflight and preparation, before
search starts. Their scoped locks last through the search phase. An explicit
memory budget also bounds the page-rounded pin request; without one, the OS
lock allowance is the limit. Logs report successful bytes and each skipped
dataset. Passing preflight does not guarantee available RAM or permission to
lock it: denial leaves storage pageable and search continues. Embedded Rust
sessions retain the explicit guard API below so the caller controls residency.

```no_run
# fn pin_example(index: &orion_ann::Index) -> orion_ann::Result<()> {
use orion_ann::{DatasetPin, PinMode, PinOptions, QuantizedKind};
let hints = index.pinning_hints()?;
println!("{hints:?}");
// For a calibrated L2-u8 cascade, pin only its quantized representation.
let residency = index.pin(&PinOptions {
    mode: PinMode::BestEffort,
    budget_bytes: 128 * 1024 * 1024,
    datasets: Some(vec![DatasetPin::Quantized(QuantizedKind::L2U8)]),
})?;
println!("{:?}", residency.report());
// Keep `residency` alive across searches. Overlapping Orion guards share locks.
# Ok(()) }
```

`PinMode::Required` returns an error and rolls back the request if the budget or
OS prevents locking. `Off` performs no locking. A selection must belong to this
index's search requirements; selecting an unprepared representation returns
`InvalidState` instead of allocating it. An empty selection locks nothing.
The guard borrows storage, so Rust prevents dropping or mutably recalibrating
the index while it is held. Pinning does not invalidate calibration or affect
cache identities, snapshot manifests or search results. Apply the desired
residency policy again after opening a snapshot on another machine.

Cascade precision is validated before sidecar preparation. A binary sketch is
not a final reranker, and exact f32/native-u8 admission cannot be followed by
quantized reranking. Increasing scalar precision (u8 to u16, i8 to i16) and
refinement to exact scoring are supported; unrelated quantizers are not ordered
by their storage width. The existing dimension/metric defaults are unchanged.

## Product quantization

The `pq-l2` and `pq-ip` admission choices use the same fitted PQ dataset with
squared-L2 and negative-inner-product lookup tables respectively. Query coordinates
remain f32. PQ is trained only when requested; existing default recipes are unchanged.
PQ admissions can use exact reranking. Compression ratio alone does not establish
an ordering against scalar quantizers, so PQ followed by u16 reranking is rejected.

The canonical recipe trains 256 centroids per subspace with a deterministic sample
of at most 4,096 rows and 12 Lloyd iterations. It uses `ceil(dimension / 8)` subspaces,
capped at 64. Standalone `PqBuildOptions` allow explicit budgets and chunk counts.
Training, code generation and page placement belong outside search timing.

Codebooks are best-effort pinned in RAM when prepared. This is not an L1/L2 residency
guarantee; `PqDataset::codebook_pin_error()` reports an OS lock refusal. Codes are
owned or mapped according to the allocation-time `storage_options` plan.
`Index::set_pq_code_storage(PqCodeStorage::OnDemand)`
can move prepared codes into a private temporary-file mapping; the OS then faults
code pages as needed. This needs temporary disk space equal to the code payload,
and transition may temporarily retain both representations. It does not map a mutable
cache path. `Resident` switches back to owned memory; use the ordinary scoped pin API
with an explicit PQ selection if codes must remain in RAM. Placement does not change
scores, calibration or snapshot identity.

```no_run
# fn placement(index: &mut orion_ann::Index) -> orion_ann::Result<()> {
// Prepare a PQ recipe by calibration, or open an existing PQ snapshot first.
index.set_pq_code_storage(orion_ann::PqCodeStorage::OnDemand)?;
# Ok(()) }
```

## Optional scores and explicit preparation

`SearchOptions::new(k, beam)` returns IDs only. Add `.with_scores(true)` to
request full-precision scores for the selected candidates. Squared-L2 is lower
is better; inner product and cosine similarity are higher is better. Internal
negative dot products and quantized admission distances are not exported as
business scores. Exact rerank scores are reused when their metric matches.

Scores preserve the selected IDs and order. Approximate final ranking can leave
the exact scores out of order; exact scores do not imply exact global top-k.
Floating-point results can differ slightly across SIMD backends. Only the
`counts[row]` prefix is valid; score padding is NaN, ID padding is `u32::MAX`.
An unrepresentable/non-finite score returns an error rather than invalid JSON.

Use `calibrate_many(samples, &[10, 100], None)` to explicitly prepare several k
values. Inspect `calibrated_ks()` and `calibration(k)` without doing any work.
Missing preparation remains `InvalidState`, including empty query batches.
Calibration records are published together after successful batch preparation;
previous records remain intact on failure, although sidecars may have been
materialized. Calibration is sample-dependent and is not a recall guarantee.

## Business identities

```rust
use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions};
# let root = std::env::temp_dir().join(format!("orion-ids-doc-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
let base: Vec<f32> = (0..32 * 8).map(|i| (i % 101) as f32).collect();
let business_ids: Vec<u64> = (100_000..100_032).collect();
let mut index = Index::new_with_ids(BaseInput::from_f32(&base, 8)?,
    IndexOptions { cache_dir: root.join("work"), threads: 1, ..Default::default() }, business_ids)?;
index.calibrate_many(base[..8 * 4].to_vec(), &[3, 5], Some(16))?;
assert_eq!(index.calibrated_ks().collect::<Vec<_>>(), [3, 5]);
let result = index.search(QueryBatch::from_f32(&base[..8], 8)?,
    &SearchOptions::new(3, 16).with_scores(true))?;
let count = result.counts[0] as usize;
println!("business IDs: {:?}", &result.external_ids.as_ref().unwrap()[..count]);
println!("scores: {:?}", &result.scores.as_ref().unwrap().values[..count]);
# drop(index);
# std::fs::remove_dir_all(root)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

IDs must be unique and match the indexed prefix length. All `u64` values are
valid business IDs, including `u64::MAX`; counts define padding. Strings remain
application-owned. The graph and ground-truth evaluation keep internal `u32`
IDs. The mapping costs eight resident bytes per point, plus a temporary sorting
copy during validation. It is stored as a separately authenticated immutable
object and does not change graph cache identity.

Python accepts `external_ids=` in `Index.from_file` and `Index.from_array`;
`SearchResult.external_ids` is a `uint64` array or `None`. CLI accepts
`--external-ids ids.json` (one JSON array in base-row order) and `--scores`.
JSONL results contain internal `ids` plus optional `external_ids`, `scores` and
`score_metric`. Large JSON u64 values must be read with a lossless integer parser,
not converted to JavaScript Number.

A complete file-based example is available as `business_search`:

```text
cargo run -p orion-ann --release --features file-queries --example business_search -- \
  base.fvecs query.fvecs business-ids.json index-options.json snapshot-dir
```

It uses the caller's `IndexOptions` JSON and u64 ID array, prepares k=10 and
k=100 (requiring at least 100 base rows), searches with scores, saves and reopens,
and checks the returned identities and scores. Existing graphs are reused when
the options and base identity match the working cache.
