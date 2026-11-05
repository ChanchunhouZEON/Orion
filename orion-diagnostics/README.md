<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# orion-ann-diagnostics

Opt-in diagnostics for Orion's production search loop. The default `native`
feature provides `RecordingObserver` and depends on `orion-ann-core`. Orion itself
only defines observer hooks and does not depend on this crate.

Disable default features to use the versioned trace schema, validation and
`TraceReplay` without the native engine. This is the configuration used by
`orion-ann-wasm`. Shared errors come from `orion-ann-error`; no second Error
implementation is defined here.

## Recording

Use `orion_ann::Index::trace_search(query, k, beam, adaptive, limits)`.
The call accepts exactly one logical f32 query row, normalizes it like ordinary
search, and requires calibration for adaptive search. Native-u8 indexes require
exact byte-valued coordinates. It returns a `SearchTrace`, including final IDs.

Events contain the scored entry, expanded nodes and deduplicated candidates,
peel/exploration/converged phases, prefilter counts and thresholds, admitted
candidates with admission scores, beam snapshots after updates, and rerank count.
Prefilter-rejected distances and final reranked distances are not recomputed.
Admission is distinct from being retained in the beam or selected as a result.
Detailed trace collection is not a QPS timing path.

`TraceLimits` defaults to 2048 events and 32768 payload items. The hard maxima are
8192 events and 65536 items. An item is a retained candidate or ID; k final-result
slots are reserved before recording. Metadata labels are bounded to 64 bytes.
These limits bound retained records, not an exact process-RSS ceiling.

When a budget is exhausted, recording stops at an event boundary. Subsequent
payloads are not copied. Search continues normally, and the returned object has
`truncated: true`, the real termination reason and the full result IDs. It never
presents a partial event prefix as a complete traversal.

## Replay and compatibility

`SearchTrace::to_json/from_json` uses schema version 1, explicit non-finite score
spellings, and an 8 MiB JSON input/output cap. Invalid versions, IDs, shapes,
counts and limits are rejected with structured errors. Validation checks the
record structure; it does not authenticate a trace or prove its provenance.

`TraceReplay::next_events(1..=1024)` returns a borrowed page. `seek` addresses an
event offset, not a graph hop, and may seek to the end. Failed cursor operations
leave the cursor unchanged. Replay does not execute search or reconstruct a
complete graph/queue state; consumers apply events or restart from offset zero.

Publication remains on hold until peer review completes.
