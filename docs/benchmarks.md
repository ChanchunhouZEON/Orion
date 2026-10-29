# Benchmark evidence

[README](../README.md) · [Historical figures](historical-benchmarks.md)

Documentation refresh: **2026-10-29**. No experiments were rerun for this refresh,
and no paper figures or existing visualization data were regenerated. The
[JSON snapshots](results/2026-10-29/README.md) retain comparison values, brackets,
provenance and source-file hashes from the completed experiment directories.

## Read the comparison method first

A source graph and a search engine are different variables. Orion can search a
PA-derived graph or a Rust-built graph; that does not turn it into either
baseline engine. Equal R/L/alpha does not establish identical topology.

Reported QPS excludes index loading/construction and calibration. Recall is
calculated after search. Each engine retains its benchmark's timing and cache
handling; eviction is best effort and cannot guarantee identical cache state.
Three trials summarize run variability but do not constitute a significance
test. Compare k, base order/prefix, query/GT files, CPU affinity, SIMD flags,
THP coverage and source graph before attributing a difference to an optimization.

The tables below use two distinct methods:

- **Interpolated same recall:** interpolate QPS on measured frontiers, without
  extrapolation. The result is an estimate, not a new measurement. JSON retains
  brackets where available.
- **Measured threshold:** choose the fastest measured median QPS whose recall
  meets a target. Actual recalls can differ and are shown explicitly.

## Local ARM: k=100

Study `six-datasets-k100-20261027`; source commit
`80054cf42be3ccc1b863884efb715e681d5756c8`; native Apple Silicon/macOS, 8 threads,
3 trials per parameter. The study manifest did not capture the exact CPU model.
Both engines were rerun; each loaded once per dataset. Orion used PA-derived
graphs, f32 resident bases, near=32, burst=32, trigger=8, ahead=4 vectors and
rerank factor=2. Rust 1.92 with repository AArch64 dot-product flags was used.

All GT files contained exactly 100 identical IDs per query, so ParlayANN's
extension beyond top-k added no IDs for k=100. Orion had 19 L points; ParlayANN
had 34 distinct Q/limit combinations. Its repeated final Q=1000 configuration
contributes six samples; other points have three. Linux THP is inactive here.

**Orion QPS change versus ParlayANN; interpolated same recall:**

| Dataset | R=0.95 | R=0.97 | R=0.98 | R=0.99 |
|---|---:|---:|---:|---:|
| `sift` | — | +20.99% | +6.27% | -5.90% |
| `glove25` | +58.03% | +31.43% | +21.47% | +13.75% |
| `glove100` | +23.65% | +2.16% | -13.99% | — |
| `gist` | +191.75% | — | — | — |
| `deep10m` | +11.09% | +12.19% | +8.40% | -5.50% |
| `msmarco_bert_1M` | +165.52% | — | — | — |

`—` means outside a measured range, not zero gain. SIFT's minimum measured
Orion recall was already above .95. GIST and MS-MARCO's measured ParlayANN
frontiers did not reach .97. The high-recall SIFT, DEEP10M and GloVe100 regressions
are retained rather than hidden by a panel-wide average.

Source: [ARM k100 snapshot](results/2026-10-29/arm-k100.json), including binary
hashes, per-target endpoints and historical reference comparisons. The latter
are not paired optimization controls.

## Local ARM: k=10 historical reference

Study `seven-datasets-near32-20261022` compares refreshed Orion measurements
with **historical** ParlayANN data. Eight threads, three Orion trials per L;
piecewise-linear QPS interpolation against reported recall, without extrapolation.
PA retains its original tie-aware recall, not strict ID recalibration.

| Dataset | R=.95 | R=.97 | R=.99 |
|---|---:|---:|---:|
| `sift` | +31.66% | +28.22% | +17.32% |
| `glove25` | +50.90% | +44.12% | +30.65% |
| `glove100` | +33.56% | +30.28% | +19.84% |
| `gist` | +45.54% | +59.88% | +26.87% |
| `deep10m` | +24.81% | +17.11% | +19.98% |
| `msmarco_bert_1M` | +42.67% | +36.72% | — |

These numbers are provisional reference comparisons, not a new paired engine
benchmark. In particular, historical GloVe metadata reports 1M points while the
current full set has 1,183,514; the old collector default may explain this, but
raw provenance was not verified. Some historical PA points have only one run.
Do not combine these ratios with the fresh k=100 study into one claimed gain.
The separate WikiAda references and notes are in the
[k10 historical snapshot](results/2026-10-29/arm-k10-historical.json).

## SIFT1B on x86

Study `20261027-165719-orion-parlay-thp-8t-n2qinxh_`; Xeon Platinum 8358P,
Ubuntu 22.04/glibc 2.35, eight physical cores (CPU IDs 0–7), k=10, three trials
per parameter. Each engine loaded once, Orion followed by ParlayANN. Orion used
native-u8 data, a PA-derived phased graph and the AVX-512 THP binary identified
by SHA-256 in the snapshot. A matching source commit was not recorded in that
snapshot; the binary hash identifies the deployed artifact.

The captured Orion process had 499,998,720 KiB of `AnonHugePages` out of
500,011,676 KiB anonymous memory: **99.9974% observed coverage**. This is a
measurement for that process, not a guarantee made by `madvise`.

**Fastest measured point meeting each target, without interpolation:**

| Target | Orion recall | Orion QPS | PA recall | PA QPS | Change |
|---:|---:|---:|---:|---:|---:|
| 0.800 | 0.813300 | 71,844 | 0.827290 | 72,414 | -0.79% |
| 0.900 | 0.912790 | 42,369 | 0.906190 | 46,982 | -9.82% |
| 0.950 | 0.957990 | 33,897 | 0.950330 | 33,086 | +2.45% |
| 0.960 | 0.963940 | 30,746 | 0.961790 | 27,459 | +11.97% |
| 0.970 | 0.972080 | 26,072 | 0.974890 | 21,616 | +20.61% |
| 0.980 | 0.980650 | 20,916 | 0.982190 | 17,472 | +19.71% |
| 0.990 | 0.990960 | 12,788 | 0.990950 | 11,330 | +12.87% |
| 0.995 | 0.995760 | 7,015 | 0.995450 | 6,901 | +1.66% |

Orion uses ID recall; PA uses the original tie-aware GT calculation. These are
reported-recall threshold comparisons, not strict identical-recall measurements.
The geometric mean gain across .95/.96/.97/.98/.99 targets is +13.33%, with
actual recalls differing. The .90 target favors PA. Timing uses each original
single-pass protocol, without a minimum-one-second repeat loop; node/run
variation remains a potential confound. DiskANN was not rerun in this paired
study, so it does not establish a current three-engine ranking.

Source: [SIFT1B snapshot](results/2026-10-29/sift1b-x86.json), with all retained
median points and page-accounting totals.

## Same engine, different graph sources

Study `graph-source-k10-k100-20261029`, same Orion binary/commit as the local
k=100 study, Darwin 25.5.0 arm64. Six datasets × two sources × two k values;
8 threads, 3 trials per L, 24 sweeps and 648 measured parameter points. Each
source/k combination loaded once. Search cascade, input files, construction
R/L/alpha/extra and prefetch settings matched within each dataset. Calibration
was fitted independently per graph/k using the same settings.

**Rust graph QPS change versus PA graph QPS in Orion; interpolated same recall:**

| Dataset | k=10, R=.95 | k=100, R=.95 | k=10, R=.99 | k=100, R=.99 |
|---|---:|---:|---:|---:|
| `sift` | -20.24% | — | -29.72% | -30.28% |
| `glove25` | +1.19% | +0.39% | -0.99% | -1.52% |
| `glove100` | -16.27% | -26.00% | — | — |
| `gist` | -58.15% | -63.66% | — | — |
| `deep10m` | -23.49% | -18.87% | -29.01% | -37.97% |
| `msmarco_bert_1M` | -67.21% | -71.90% | — | — |

Here `diskann` in the JSON labels **Rust graph source**, not DiskANN search.
PA usually wins; GloVe25 is near parity. Rust did not reach some high-recall
targets covered by PA. A missing ratio is not an extrapolated loss.

Four Rust graphs and all six PA graphs reused historical caches. DEEP10M and
MS-MARCO Rust graphs were built for this study. Historical build versions,
seeds and internals were not controlled, so the comparison describes available
graphs rather than an isolated algorithmic effect. MS-MARCO is raw MIPS while
the current Rust builder uses L2, adding a construction-objective confound.

Used quantization/projection sidecars were SHA-256 checked. GloVe25 initially
had 15 differing bytes; its experimental Rust sidecar was aligned with PA and
all four GloVe25 sweeps rerun. The final audit reports identical used sidecars
and 43 original cache files unchanged. Only final aligned results appear here.

Source: [Graph-source snapshot](results/2026-10-29/graph-sources.json), including
exact interpolation brackets, measured threshold points, recall ranges and audit.

## Reproduction

Use the commands and environment in the [README](../README.md#reproducing-a-sweep),
with explicit graph source. Run k=10 and k=100 separately, recording
`--print-config` and binary hashes. Dataset path identity and historical caches
can vary between machines; a fresh graph need not reproduce a cached graph's
exact frontier. Snapshot hashes identify the supplied analyses, not a promise of
bitwise identical performance on another system.

Keep load/build timings separate from search QPS. Retain actual recalls,
per-trial QPS, CPU placement and page coverage rather than just a headline ratio.
For a new baseline comparison, run engines under a declared shared protocol and
record any engine-specific timing/recall behavior.
