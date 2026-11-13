<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Labels and filtered search

Labels are the first independently mutable metadata in an `Index`. They do not
modify vectors, graph edges, quantized sidecars or calibration. The initial
filtered-search implementation **exhaustively scans the allowed subset**. It is
a bounded correctness baseline, not a filtered ANN performance claim.

## Business IDs and internal rows

`LabelUpdate.add` and `remove` contain **internal u32 row IDs**, exactly the IDs
in `SearchResult.ids`. They do not accept business u64 IDs. Applications using
business IDs should pass `ExternalLabelUpdate` to `update_labels_external`.
The runtime resolves the entire batch in one scan of the existing mapping,
without keeping a second N-entry hash map, and publishes through the same
transaction as internal-ID updates. Unknown add **or remove** IDs fail the
whole batch; no earlier change is published. The revision check at publication
also rejects a concurrent writer's stale update. An index without a business
mapping rejects this method, even for an empty batch.

`resolve_external_ids` remains available when internal rows are needed for other
operations. It preserves input order. The mapping is immutable for the lifetime
of an Index; neither label update method changes it.

```no_run
use orion_ann::{Index, LabelKey, ExternalLabelUpdate, LabelFilter, FilterSearchOptions, QueryBatch, Result};
fn example(index: &Index, queries: &[f32]) -> Result<()> {
    let book = LabelKey::new("category", "book")?;
    let movie = LabelKey::new("category", "movie")?;
    let business_ids = vec![1001, 1002];
    let revision = index.update_labels_external(index.label_revision()?, &[
        ExternalLabelUpdate { label: book.clone(), add: business_ids.clone(), remove: vec![] },
    ])?;
    let options = FilterSearchOptions::new(10).with_scores(true);
    let filter = LabelFilter { all: vec![book.clone()], ..Default::default() };
    let results = index.search_filtered(QueryBatch::from_f32(queries, index.dimension())?, &filter, &options)?;
    assert_eq!(results.label_revision, revision); // if this example owns the only writer
    // Both changes become visible together; a failed batch publishes neither.
    index.update_labels_external(revision, &[
        ExternalLabelUpdate { label: book, add: vec![], remove: business_ids.clone() },
        ExternalLabelUpdate { label: movie, add: business_ids, remove: vec![] },
    ])?;
    Ok(())
}
```

## Request semantics

`LabelKey` is an exact, case-sensitive namespace/value pair. Neither string may
be empty; their combined UTF-8 length is at most 1024 bytes. There is no implicit
case folding, Unicode normalization, numeric range or expression evaluator.

A `LabelFilter` means `ALL(all) AND ANY(any) AND NOT ANY(none)`:

- Empty `all` or `none` imposes no restriction. Empty `any` also imposes no restriction.
- An entirely empty filter allows the entire index, subject to the scan budget.
- Unknown labels represent empty sets: an unknown `all` term gives zero results;
  unknown `any` terms contribute no matches; an unknown `none` term excludes nothing.
- A filter accepts at most 4096 terms. One filter snapshot serves the complete query batch.

`FilterSearchOptions` is separate from adaptive `SearchOptions`. No calibration
is needed for this exact path. `max_candidates` defaults to 100,000 and rejects
larger matching sets **before vector evaluation**; increase it explicitly to
accept a longer scan. This bound does not bound the cost of resolving labels or
all output allocations. Exact scan uses original-metric scoring, O(k) heap space
per query worker, and ascending internal ID for equal representable scores.
It does not use the quantized cascade or change its SIMD/prefetch dispatch.

Only allowed IDs can be returned. `counts` describes valid prefixes when fewer
than k rows match; ID/score/business-ID padding has the same meaning as ordinary
search. Optional scores use squared L2, unnegated inner product or cosine
similarity. Zero-norm cosine input and unrepresentable scores are rejected.

## Publication and concurrent reads

`update_labels` and `update_labels_external` use optimistic version checking
and serializes writers. Initial revision is zero; a successful state change
increments it once. A no-op leaves it unchanged. Duplicate IDs in one add/remove
list are idempotent. Overlapping add/remove IDs, duplicate labels in a batch,
invalid IDs or stale revisions reject the entire batch.

Each query clones a reference-counted immutable label snapshot under a short
lock. Updates build new membership lists and publish the complete map once;
unmodified lists are shared. Neither membership lookup nor vector scoring holds
the publication lock. Existing queries continue with their original revision.
This is query-snapshot consistency, not cancellation of in-flight queries after
permission revocation. Atomicity is local to this Index instance, not a distributed
transaction across independently opened processes.

The prototype bounds a live state to 4096 labels and 1,000,000 total memberships;
a batch accepts at most 1,000,000 supplied IDs. Sorted membership arrays avoid a
full N-bit mask per label. Updating a large label still copies that label's list,
and readers may retain previous versions. These limits do not guarantee a fixed
process RSS or include labels in the graph/base preflight estimate.

A label object is bound to the base content identity, indexed row count and
business-ID mapping identity. Equal point counts alone do not allow reuse. A
newly built Index starts with empty labels even if it reuses a working graph
cache. Restore a snapshot to retain labels. Independent copies restored from
one revision may diverge; revision numbers are not global transaction IDs and
must not be used alone as cross-instance cache keys.

## Saving and restoring

An update acknowledgement means **in-memory visibility**. Explicit `Index::save`
provides a durable checkpoint. Drop does not implicitly save, and there is no WAL
or automatic recovery of updates after the last checkpoint.

Save captures one label revision, writes a SHA-256-addressed immutable label
object, then publishes a manifest referencing it. Graph/base/sidecar objects are
reused. Concurrent label updates cannot mix revisions inside a saved object.
A failed save leaves the live state and prior snapshots usable; an unreferenced
immutable object may remain if manifest publication fails.

New snapshots use format v2, so old readers reject them rather than silently
ignoring labels. Readers still accept unlabelled v1 snapshots; attaching a label
object to a v1 manifest is rejected. Open validates the label digest, format,
identity, revision, membership ordering and bounds. Bundles include label objects
and retain their meaning after relocation. Never mutate a cache object in place.

Run the complete Rust demo with:

```sh
cargo run -p orion-ann --example label_updates -- /tmp/orion-label-demo
```

Python exposes `LabelKey`, `LabelUpdate`, `ExternalLabelUpdate`, `LabelFilter`,
`update_labels`, `update_labels_external`,
`resolve_external_ids`, `label_revision` and `search_filtered` with the same
runtime contracts. Input arrays retain the existing copy-before-GIL-release
behavior. Filtered results expose `result` and `label_revision`.

For an Index built with `external_ids`, the business-ID path is:

```python
from orion_ann import ExternalLabelUpdate, LabelKey, LabelFilter

book = LabelKey("category", "book")
revision = index.update_labels_external(index.label_revision, [
    ExternalLabelUpdate(book, add=(1001, 1002)),
])
results = index.search_filtered(queries, LabelFilter(all=(book,)), k=10, scores=True)
# results.result.external_ids holds business IDs; results.label_revision
# identifies the immutable label state used for every query in this request.
```

For CLI search, save labels using Rust or Python and provide the label filter as
JSON (`{"all":[{"namespace":"category","value":"book"}]}`):

```sh
orion --snapshot saved-index --base base.fbin --query query.fbin --k 10 \
  --label-filter filter.json --filter-max-candidates 100000 --scores
```

The CLI does not update labels. It reads one saved state, reports `filtered_exact`
and the label revision, and does not require or create adaptive calibration.
Ground truth, when supplied, must describe the same allowed subset.
