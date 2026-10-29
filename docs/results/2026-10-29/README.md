# Documentation result snapshots

These small JSON files are derived from completed experiments and support
[the benchmark guide](../../benchmarks.md). They do not modify the original
experiment outputs, visualization JSON, figures or paper datasets.

| File | Evidence |
|---|---|
| [arm-k100.json](arm-k100.json) | Fresh six-dataset Orion/PA k=100 comparison, 2026-10-27 |
| [arm-k10-historical.json](arm-k10-historical.json) | Refreshed Orion versus historical PA, 2026-10-22; provisional metadata |
| [graph-sources.json](graph-sources.json) | Same Orion on PA/Rust graphs, k=10/100, 2026-10-29 |
| [sift1b-x86.json](sift1b-x86.json) | Paired 8-thread SIFT1B Orion/PA run, 2026-10-27 |

Each snapshot includes its source study identifier and source analysis SHA-256.
Binary/commit identities are included where recorded. Absolute workstation
paths have been removed from derived metadata. These are selected summaries,
not full raw-log archives; preserve the original study directories for audits.
`null` means a target lies outside the measured range or a quantity was not
recorded. Interpolated values must not be labeled measured QPS.
