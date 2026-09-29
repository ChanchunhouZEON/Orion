
[Back to the main README](../README.md#table-of-contents)

This document collects the convergence-side and cascade-stage ablations,
adaptive-search experiments, and matched-state diagnostics. Run all reproduction commands
from the repository root; see [Quick Start](../README.md#quick-start) for setup.

## Table of contents

- [Convergence-side Ablation](#convergence-side-ablation)
- [Cascade-Stage Ablation](#cascade-stage-ablation)
- [Adaptive-search diagnostics](#adaptive-search-diagnostics)
- [Adaptive switching and early-exit ablation](#adaptive-switching-and-early-exit-ablation)
- [Matched-state diagnostics](#matched-state-diagnostics)

## Convergence-side Ablation

![Convergence ablation — GIST](../visualizations/ablation_gist.png)

A clean **2×2 factorial** on the two convergence-side knobs (extras × early-stop), **all four variants on the same cascade, the same PA-built base graph, and one shared `calibrate()`**:

| variant | extras | early-stop | what it isolates |
|---|---|---|---|
| **Origin**         | off | off | bare-cascade reference point |
| **No extras**      | off | on  | marginal value of early-stop alone |
| **No early-stop**  | on  | off | marginal value of extras alone |
| **Full**           | on  | on  | both knobs (production) |

Extras off = the post-convergence branch falls back to walking `local + remote` (the same nav-mode path) instead of switching to `local + extra`; flag plumbed through `algorithm::search::INCLUDE_EXTRAS`. Early-stop off = `early_exit_limit = usize::MAX`; the beam runs to PQ exhaustion. Both flags are pure search-time — no graph rebuild — so every variant calls the same `search_compose` on the same graph with only two booleans changed.

**Earlier versions of this panel had two conflations** that are worth flagging because the conclusions shifted twice:
- v1 used Microsoft Vamana as origin — conflated "no cascade backbone" with "no convergence machinery," producing a 4-10× gap that read as a convergence-side win but was really the cascade kernels.
- v2 built a separate `max_extra=0` graph for "no extras" — conflated build-time topology (PA's 2-pass Vamana vs Rust's 1-pass) with the search-time extras contribution, producing a fake "extras +2-3pp recall ceiling" lift.

This v3 design eliminates both. The cascade-vs-no-cascade headline now lives in the 3-engine sweep panel where it belongs.

**Results at L=256** (full L sweep in the PNG/PDF; v3 numbers, single PA-built graph, shared calibration):

| Dataset | default cascade | Origin (R / QPS) | Full (R / QPS) | No-extras Δ vs Origin | No-EE Δ vs Origin |
|---|---|---|---|---|---|
| **SIFT 1M**         | `none → l2-u8 → f32`      | 0.999 / 46 K | 0.999 / 65 K | -0.000 R, +35 % QPS | +0.000 R, +12 % QPS |
| **GIST 1M**         | `jl → l2-kt → f32`        | **0.992** / 16 K | 0.986 / 20 K | -0.001 R, +13 % QPS | -0.003 R, +8 % QPS |
| **GloVe-25**        | `none → mips-i8 → ip-f32` | 0.999 / 64 K | 0.998 / 84 K | -0.000 R, +23 % QPS | +0.000 R, +7 % QPS |
| **GloVe-100**       | `none → mips-i8 → ip-f32` | **0.962** / 37 K | 0.960 / 42 K | +0.000 R, +11 % QPS | -0.001 R, +6 % QPS |
| **Deep10M**         | `none → l2-u8 → f32`      | **0.996** / 41 K | 0.994 / 52 K | -0.001 R, +18 % QPS | -0.001 R, +5 % QPS |
| **msmarco_bert_1M** | `none → mips-i8 → ip-f32` | **0.968** / 26 K | 0.966 / 30 K | -0.001 R, +6 % QPS | -0.001 R, +8 % QPS |

(Bold = highest R in the row. Δ vs Origin shows how each individual knob shifts the trade-off from the bare-cascade reference.)

- **Both knobs trade recall for QPS in the same direction**, with very small recall costs (≤ 0.6 pp) and modest QPS gains (5–35 %). The convergence-side machinery is a *trade-off lever*, not a recall-improving feature. Production picks "both on" (Full) because the trade is consistently in our favor — small recall hit for non-trivial QPS.
- **The cascade itself does the heavy lifting.** All four variants overlap tightly across every dataset's QPS-vs-recall curve (visible in the figure). What the convergence knobs change is *where on the same Pareto frontier* you sit, not the shape of the frontier.
- **Origin has the highest recall ceiling on 4 of 6 datasets** (GIST, GloVe-100, Deep10M, msmarco). This is intuitive given the design: `ee=MAX` runs the full beam, so deeper exploration; `INCLUDE_EXTRAS=false` keeps the post-convergence walk on `local+remote` (long-range shortcuts) rather than switching to `local+extra` (refinement). The remote shortcuts beat the extras zone for recall at the tail. SIFT and GloVe-25 are the exceptions — both nearly saturate at R≈1.0 regardless of variant, so the recall ceiling difference is below measurement noise.
- **GIST has the largest trade-off cost**: Full vs Origin is +25 % QPS for -0.6 pp recall. On every other dataset the cost is < 0.1 pp recall.
- **Notable for design**: on the same graph, the rerank-mode mode-switch from `remote` to `extra` post-convergence doesn't help recall (and on GIST mildly hurts it). The extras zone's value, if any, lives in **build-time** behavior — does keeping pruned-loser candidates change the local-zone shape? That'd be a `PA ex16` vs `PA ex0` comparison (same builder, different cap), outside this ablation's scope.

## Cascade-Stage Ablation

![Cascade ablation — msmarco_bert_1M](../visualizations/cascade_ablation_msmarco_bert_1M.png)

Companion to the convergence ablation above. That one isolates the **convergence-side** machinery (early-stop / extras) on a fixed cascade; this one isolates the **three cascade tiers** (prefilter / admission / rerank) on the production graph. Four curves per dataset:

- **Full** — per-dataset default cascade (e.g. `none → l2-u8 → f32` for SIFT, `jl → l2-kt → f32` for GIST).
- **No prefilter** — `PrefilterChoice::None`, keep admission + rerank.
- **No rerank** — `RerankChoice::None`, top-k from admission PQ ordering directly.
- **Admission only** — both auxiliary tiers off; just the PQ-ranked admission distance.

All four share **one built graph and one `calibrate()` call** so QPS deltas are attributable to the cascade change, not to build/thermal/calibration drift. The driver loads the cached PA graph (`cache/orion_parlayann/<ds>_n…_pct60.pgraph`) in 0.1–2 s — same artifact the convergence ablation and the `orion` bin produce — so QPS reflects the steady-state production path rather than a first-launch rebuild.

| Dataset | default cascade | rerank lift (max R) | rerank speedup at iso-recall | prefilter speedup at iso-recall |
|---|---|---|---|---|
| **SIFT 1M**         | `none → l2-u8 → f32`     | **+1.8 pp** (0.999 vs 0.981) | ~3.8× at R=0.98 | n/a |
| **GIST 1M**         | `jl → l2-kt → f32`       | **+5.0 pp** (0.986 vs 0.937) | ~1.7× at R=0.92 | **~2.3× at R=0.95** |
| **GloVe-25**        | `none → mips-i8 → ip-f32` | **+6.7 pp** (0.998 vs 0.931) | ~5.6× at R=0.93 | n/a |
| **GloVe-100**       | `none → mips-i8 → ip-f32` | **+3.9 pp** (0.960 vs 0.921) | ~1.8× at R=0.92 | n/a |
| **Deep10M**         | `none → l2-u8 → f32`      | **+3.5 pp** (0.994 vs 0.959) | ~3.2× at R=0.96 | n/a |
| **msmarco_bert_1M** | `none → mips-i8 → ip-f32` | **+13.3 pp** (0.966 vs 0.833) | **~6.6× at R=0.83** | n/a |

- **Rerank dominates on IP / quantization-sensitive workloads.** The msmarco panel shows two cleanly separated populations — `full`/`no-prefilter` overlap at the top, `no-rerank`/`admission-only` overlap at the bottom — because i8 MIPS quantization loses critical ranking signal on raw-IP BERT vectors with their wide L2-norm spread; the f32 IP pass re-orders the top-`k × rerank_factor` (=20) from "wrong" to "right." At iso-recall R=0.83 the full cascade is **6.6× faster** than admission-only.
- **Prefilter only matters when admission is expensive.** Only GIST (960D, `l2-kt` admission ≈ 480 i8 ops per node) sees prefilter benefit — the JL Hamming pre-rejection shields rerank from a much larger candidate pool, delivering a **2.3× speedup at R=0.95**. Every other dataset in the panel correctly defaults to `PrefilterChoice::None` because the i8/u8 admission distance is already cheap enough that pre-filtering loses more in setup than it saves in pruning.
- **The rerank cost is asymmetric in L.** At very low L (= 16 on GIST), `no-rerank` actually beats `full` on QPS (140 K vs 116 K) because the rerank stage has fixed per-search overhead the small beam can't amortize. The cascade only pays off above R ≈ 0.85; below that, simpler is faster.

**Reproduce both panels in one shot.** `DATASETS="sift gist glove25 glove100 msmarco_bert_1M" bash benchmark/scripts/run_ablations.sh`. The unified driver auto-resolves `$PY` via `benchmark/scripts/_env.sh` (override with `PY=/path/to/python`), builds the binary, then runs `--algorithms ablation` and `--algorithms cascade-ablation` per dataset back-to-back. Cache-hits load each graph in ~0.1–2 s; cache-miss paths rebuild + write the cache for next run. Set `ALGORITHMS=cascade-ablation` (or `ablation`) to skip a panel. Renders both `ablation_<ds>.{png,pdf}` and `cascade_ablation_<ds>.{png,pdf}` via the matching `plot_*.py` scripts; per-dataset panels for the four datasets not shown above sit alongside in `visualizations/`.

## Adaptive-search diagnostics

The `adaptive` executable compares three post-convergence policies on the
same index with early exit disabled: `full_neighbor` (local + remote),
`local_only`, and `local_extra`. All use the same cascade, calibration,
navigation prefix, convergence detector, and convergence-dependent PQ flush
schedule. This isolates neighbor selection, not every difference from a
conventional beam implementation.

```bash
cargo run --release -p benchmark --bin adaptive -- sift \
  --graph-source parlayann --k 10 --search-list-sizes 48,100,200,256 \
  --threads 8 --trials 5 --diagnostic-queries 1000
```

The executable shares dataset shortcuts, path/metric/cascade overrides,
cache validation, and per-k L schedules with `orion`. Replace `sift` with
`gist` for the high-dimensional L2 cascade. Omit `--search-list-sizes` to
use the configured schedule. A ParlayANN cache miss requires a matching
staged export. `--print-config` resolves inputs without building the index.

Results default to `visualizations/adaptive_<dataset>_k<k>.json`; use
`--output` to keep multiple runs. Detailed instrumentation uses an evenly
spaced query sample (`--diagnostic-queries 0` selects all queries); timed
trials use all queries and a no-op observer. Trial order rotates between
arms. Existing lightweight production counters remain active in both passes.

The JSON includes per-query and mean expansions, first switch and reversals,
unique encountered objects, and separate admission/prefilter/rerank counts.
Admission counts include the entry. Prefilter threshold recomputations and
reranking can evaluate objects already encountered, so total stage NDC is
not the unique-object count. First discovery is before prefilter; first
admission means passing the admission cutoff, not necessarily surviving
the subsequent PQ merge. Entry is step 0 and missing targets are `null`.
Paired discovery means use only the same query/target pairs discovered by
both arms; discovery fractions and final recall are reported separately.

`--recall-targets 0.9,0.95,0.99` reports the best measured QPS reaching each
target, together with its actual recall and L. An unreachable target is
`null`; no extrapolated or interpolated dominance is claimed. These sweeps
do not implement matched-state branch replay or validate pruning witnesses.

Visualize a completed run with the shared chart style:

```bash
python visualizations/plot_adaptive.py visualizations/adaptive_sift_k10.json
# Optional: choose the detailed-panel L and output directory.
python visualizations/plot_adaptive.py visualizations/adaptive_gist_k100.json \
  --detail-l 256 --out-dir visualizations/adaptive_figures
```

Each input produces PNG/PDF overview, per-L stage/phase/discovery panels,
paired discovery-step differences with common-target counts, and (when
available) measured QPS at common recall targets. Inputs are never merged.
The detail panel defaults to the largest measured L. Unreached targets
are marked explicitly; discovery curves keep unfound targets in the
denominator. Multiple JSON paths may be passed to render several runs.

## Adaptive switching and early-exit ablation

`adaptive_ee_ablation` runs a same-graph 2x2 experiment: full neighbors,
adaptive neighbors, full neighbors with EE, and adaptive neighbors with EE.
Every arm uses the same cascade, calibrated admission threshold, and
convergence-dependent flush scheduling. The two EE arms share one calibrated
EE limit; `--ee-limit` can override it for controlled experiments.

```bash
cargo run --release -p benchmark --bin adaptive_ee_ablation -- sift \
  --graph-source parlayann --k 10 --threads 8 --trials 5 \
  --diagnostic-queries 0 --recall-targets 0.9,0.95,0.99 \
  --output visualizations/adaptive_ee_ablation_sift_k10.json
```

Diagnostics default to all queries and are separate from QPS timing. The JSON
records per-query NDC before/after the **first** converged expansion, with any
later reversals included in the tail. Entry scoring belongs to the prefix;
final rerank is counted separately. Stage counts include visited-set-deduplicated
candidate evaluations and repeated prefilter threshold work, not equal-cost
full-precision operations. `early_exited` records an actual EE-triggered break.
Observer/no-observer results and the shared pre-convergence prefix are checked
at runtime.

For each recall target, `matched_recall` reports the lowest measured NDC and
highest measured QPS configurations separately, including actual recall and L.
These meet a minimum recall rather than exactly matching recall; an unreachable
target is `null`. Existing output files are never overwritten. The experiment
tests complementarity without presuming a positive interaction.

## Matched-state diagnostics

`matched_state` replays three deterministic full-neighbor prefixes and checks
logical state equality immediately before the first converged expansion collects
neighbors. It then follows full, local-only, and local-plus-extra policies with
EE disabled. It measures no QPS and never overwrites existing output files.

```bash
cargo run --release -p benchmark --bin matched_state -- sift \
  --graph-source parlayann --k 10 --diagnostic-queries 200 \
  --diagnostic-ls 64,128,256 --track both --geometry-alpha 1.1 \
  --output visualizations/matched_state_sift_k10.json
```

This diagnostic currently accepts L2 only. `exact` uses full-precision L2
admission without prefilter/rerank; `cascade` uses the resolved production
recipe. Both use the same calibration rule/threshold, but their checkpoints
are not asserted to match each other. `--diagnostic-ls` controls this binary's
L values independently of the performance sweep. Queries run sequentially.

The JSON contains checkpoint beam/pending state, candidate-zone audits,
branch NDC/recall, actual post-switch PQ retention, and first discovery times.
Geometric audits use Euclidean distances: `--geometry-alpha` is explicit and
is not automatically equated to a builder's potentially squared-distance
alpha. Witnesses are existential local anchors, not historical pruning
provenance. Coverage is tested only for ground-truth targets missing from the
checkpoint beam; grouped summaries use those inside its exact enclosing ball.
Empty coverage sets are not counted as successes. No-switch queries remain
in the output.

Audit distance computations are recorded separately from search-stage NDC.
Admission-cutoff eligibility ignores the prefilter; later PQ retention is
observed in the full continuation rather than inferred from the cutoff.
Full-only returned targets indicate branch disagreement, not causal
attribution to an individual remote edge. These sampled checks do not certify
global shortcut reachability, a universal candidate budget, or a theorem.
