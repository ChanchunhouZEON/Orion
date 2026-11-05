<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# Pre-migration sidecar reference

These immutable files were captured on Apple Silicon before the sidecar and
calibration migration, using the working tree based on `20babdbdcf910f123d077004c8e850b59323a705`
with the preceding padding/compressed-index work already applied.

`sidecar_migration.rs` defines the deterministic 48 × 32 input, projection seed
42, Hadamard width 64 and calibration graph/queries. It compares full serialized
files and calibration diagnostic bytes for typed and runtime-dimensional input.
There is deliberately no fixture-regeneration switch. Rotation generation uses `ln` and `cos`; their floating-point rounding depends
on both the architecture and the platform math library. Rebuilding these exact
bytes is therefore checked on macOS AArch64, where they were captured.

Every native platform independently checks byte-exact decode/encode of all ten
frozen sidecar files, and byte-exact agreement between typed and padded builds
on that host. Scalar-oracle tests also run across the supported architectures.
These checks preserve portable cache decoding without requiring independently
trained rotations to be bit-identical across operating systems.

Disk code strides remain canonical. A loader may add neutral SIMD padding in
memory without changing any fixture bytes. L2KT all-positive/all-negative range
fitting has a separate regression test because the old implementation incorrectly
included the allocator's zero tail when fitting its quantizer.
