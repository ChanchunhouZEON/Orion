Search a batch. Search uses the calibration for this k.

Files are decoded as a complete batch. For bounded memory, pass separate
ndarray batches. No recall evaluation or cache flushing occurs here.
Require 1 <= k <= point_count and k <= search_list_size <= u32::MAX/4.
Query dtype is independent of base dtype. Float coordinates must be finite;
cosine rows must be nonzero. Wide integers retain their original values
for exact scoring (the output scores are still float32). Returned
uint32 IDs preserve row order; use counts to exclude UINT32_MAX padding.
Empty batches are supported after control/calibration validation.
One or two rows search synchronously on the calling thread with the GIL
released; larger batches use the worker pool. Independent requests are
not combined. Threads do not limit external callers. Shapes are unchanged.

scores=True adds float32 scores of the returned candidates and a
score_metric label. Squared L2 is lower-is-better; inner product and
cosine similarity are higher-is-better. Scores preserve result order
and do not imply exact global top-k. Ignore NaN padding using counts.

Raises OrionError: missing calibration is ORION_INVALID_STATE; shape,
value and backend failures carry structured reports. Ordinary invalid
input can be corrected and retried. An unexpected Rust unwind instead
invalidates the native handle: reopen it before further operations.
