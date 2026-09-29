# Key Optimizations

[← Back to main README](../README.md)


### Memory Efficiency

| Optimization                                     | Impact                                                                                     |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Slab indexed by location                         | max_extra per node bounded by slab cap                                                     |
| `max_extra` parameter caps stride                | PhasedGraph stride = `HEADER(4) + max_degree + max_extra`, cache-line aligned              |
| Dataset freed before candidate extraction        | Peak memory reduced by ~50 MB on 100K datasets(linear increasing with dimension x num_pts) |
| Single-pass PhasedGraph build with stack buffers | No intermediate `Vec<Vec<u32>>` allocation                                                 |
| `AlignedBoxWithSlice<u32>` slab                  | Cache-line aligned, zero-copy reads                                                        |

### Search-Phase Optimizations

Every search request runs through a **three-axis cascade** where each axis is selected independently. The same loop body, prefetch shape, convergence detector, and early-exit checker drive every combination — the axes only swap the per-hop distance kernel and the per-vertex sidecar.

```
 prepare scratch ──► peeled warm-up ──► [Prefilter? cheap reject] ──► [Admission: quantized PQ + cadence] ──► [Rerank: f32 top-k×RF]
                       (hops 0..=2)      (every neighbour, optional)   (every visited, every hop)            (once, post-convergence)
```

CLI surface: `--prefilter <none|jl|jl-hadamard|rabitq> --admission <l2-u8|l2-u16|l2-kt|mips-i8|mips-i16> --rerank <f32|ip-f32|u16>`. Per-dataset defaults (`Cascade::default_for_specified_dimension_and_metric`, `orion-cli/src/cascade.rs`) pick the production triple automatically. **None of the per-stage optimisations depend on the metric** — the same beam loop, prefetch shape, PQ machinery, and admission datasets run whether the cascade is `none → l2-u8 → f32` or `jl → mips-i16 → ip-f32`.

Sub-sections below walk the stages in execution order; the **shared kernels** (`DistanceStream`, NEON distance functions, `Neighbor` PQ) used by every stage are factored out at the end.

#### 1. Per-query preparation — scratch reuse

A `InMemScratchPool` allocates `num_threads` `InMemSearchScratch` buffers up front, handed out through crossbeam's lock-free `ArrayQueue` (no mutex). Each query takes one, runs `prepare_for_query()` to clear PQ / seen-set / `dist_buffer` / `merge_scratch` **in place** (no realloc), and returns it. The four critical fields are sized once at construction so the steady-state hop never touches the allocator:

- **`pad16` PQ buffer alignment.** `pq.data` and `merge_scratch` capacities both rounded up to a multiple of 16 `Neighbor`s (12 B/entry → 192-B boundary = 12 NEON regs = 1.5 M2 lines). `mem::swap` between the two on every `batch_merge` without resize; `copy_within` in `insert` uses full-vector loads. Killed the L-mod-8 jitter signal (L=18 / L=22 had 3–5× the CV of L=16 / L=24). [`orion/src/model/neighbor/neighbor_priority_queue.rs:27`]
- **`dist_buffer` fixed at `MAX_GRAPH_DEGREE × MAX_FLUSH_INTERVAL = 100 × 4 = 400`.** Sized at construction; per-hop `reserve()` removed so cmov-compact's write pointer stays valid across `FLUSH_INTERVAL=4` post-converged hops. [`orion/src/model/scratch.rs:85–119`]
- **Visited-set — linear-probe, PA-aligned sizing.** Initial table = `2 × (L + 1) × max_degree` rounded to next pow-2; SIFT L=64 → 16384 slots / 64 KB, fits L1 with load < 50 %. A bucketed SwissTable variant was reverted — probe-chain false-positives at >50 % load → recall regressions. [`orion/src/model/visited_set.rs`]
- **`prepare_for_query` resets in place.** `pq.clear() + set_capacity`, `seen.resize_for` (only grows if `L` increased vs the last query), threshold-EMA fields zeroed, convergence + early-exit checkers reset. [`orion/src/model/scratch.rs:123–144`]

#### 2. Warm-up — 3-hop loop peeling

The first three hops run a dedicated **`expand_peeled_hop_l2`** (one variant per admission tier) that skips convergence / flush / 3-way merge bookkeeping. During hops 0..=2 the PQ transitions empty → partial → just-full, so none of that machinery can fire — but in the main loop it lives behind branches and pollutes the icache. Peeling them out strips ~10 instructions / vertex from the warm-up window and saves the branch-predictor's first warm-up misses. **+1.2 – 2.3× low-`L` QPS.** [`orion/src/algorithm/search/in_mem_search_l2.rs:65–261`]

#### 3. Prefilter tier (`--prefilter`)

After warm-up, every neighbour fetched off the graph hits the prefilter first (if any). The prefilter computes a **1-cache-line popcount** against a sketch of the base vector and rejects before the wider admission slab is touched.

| Option | Per-vertex sidecar | Score | Best for | impl |
|---|---|---|---|---|
| `none` | — | — | low/mid-D, hop ≤ 4 lines | — |
| `jl` | 1024-bit JL Sparse (NZ=9) + `‖v‖` (MIPS only) | popcount (L2) / popcount × `‖v‖` (MIPS) | high-D L2 / cosine (GIST, Wiki-ada) | `orion/src/model/dataset/jl_sparse_dataset.rs:115`, `:376` (MIPS) |
| `jl-hadamard` | 1024-bit FWHT sign sketch | popcount | dense-D where JL Sparse plateaus | `orion/src/algorithm/search/stage/prefilter/jl_hadamard.rs` |
| `rabitq` | RaBitQ B=1 / B=4 rotated sign-pack | popcount | sub-bit discrimination; GIST high-recall | `orion/src/algorithm/search/stage/prefilter/rabitq.rs` |

L2 / MIPS split lives in the type — `JLSparseDataset` (no norms) vs `JLSparseDatasetMips` (norms slab cache-aligned next to codes), separate so L2 pays nothing for an unused norms slab. NZ tunables are per-dataset (L2=9, MIPS=9 in the GLM-vivid sweep). [`orion/src/algorithm/search/stage/prefilter/jl.rs:35–117`, `:119–200`]

A u8-as-L2-proxy prefilter was tried and discarded for angular: on unit-norm vectors the L2² range collapses to `[0, 4]`, so 256 quantisation levels coarsen to where everything passes — bit-sign sketches are what pays.

#### 4. Admission tier (`--admission`)

Survivors of the prefilter enter the admission tier — the per-hop main work. Two pieces, both stage-uniform:

**(a) Storage — `QuantizedDataset<Q, N>` + the `QuantSpec` trait.** Spec-trait machinery bundling storage element, scaling, NEON kernel, sidecar magic, and `ALIGN_ELEMS` (rounds `STRIDE × sizeof(Storage)` up to a 32-B SIMD-aligned vertex stride). The alignment knob fixed the glove-100 unaligned-load tax — `id × 100` is not 16-byte-aligned in a packed `Vec<i8>` — for the **0.70× → 0.96× PA** single biggest jump in the angular stack. Sidecars built lazily on first `ensure_quantized_dataset_*`, memcpy-loaded thereafter (~50 ms / 1.2 M points); storage is `AlignedBoxWithSlice<Q::Storage>` at 32 B. [`orion/src/model/dataset/quantized_dataset.rs:52–138` trait, `:503–512` struct + `STRIDE`, `:619–670` sidecar I/O]

| Spec | CLI | Storage | Kernel | Score | impl |
|---|---|---|---|---|---|
| `L2U8` | `l2-u8` | u8 (1 line / vert, ¼ f32) | `L2U8Distance` (`vabdq_u8 → vmull_u8 → vpadalq_u16`) | direct L2 | `quantized_dataset.rs:199–272` |
| `L2U16` | `l2-u16` | u16 (2 lines / vert) | `L2U16Distance` | precise L2 | `quantized_dataset.rs:274–360` |
| **`L2KTDataset`** (+`MipsI8`) | `l2-kt` | i8 + i32 `‖x_i8‖²` sidecar | `IpI8Distance` (`sdot`) + 1 scalar | `‖q‖² + ‖x‖² − 2⟨q, x⟩` — kernel-trick L2 | `l2_kt_dataset.rs:74–85` |
| `MipsI8` | `mips-i8` | i8 (`127 / max\|x\|`) | `IpI8Distance` (`sdot`) | `−Σ q · x` | `quantized_dataset.rs:362–430` |
| `MipsI16` | `mips-i16` | i16 (`32767 / max\|x\|`) | `IpI16Distance` (`vmull_s16 + vpadalq_s32`) | i16 raw MIPS — used when BERT long-tail caps i8 at R≈0.97 | `quantized_dataset.rs:432–501` |

The **L2-kernel-trick** (`l2-kt`) is the headline admission optimisation on high-D L2: GIST D=960 `l2-u8` is bandwidth-bound on `vabdq_u8 → vmull_u8 → vpadalq_u16` (~14 SIMD ops / 32-B chunk); `l2-kt` collapses the kernel to `sdot` (~2 ops / 32-B chunk) and recovers L2 via the identity. `‖x‖²` is a per-vertex i32 lookup, `‖q‖²` a per-query constant. Default on `gist` + `fashion-mnist`.

**(b) Per-hop cadence — admit, flush, merge.** Each hop runs `admit → flush → merge` against the shared scratch:

- **Branch-free admission via cmov-compact.** Per-hop kernel writes every `(id, dist)` into `dist_buffer` and advances the write pointer conditionally (`w += (dist < pq_worst) as usize`). Kills the 50/50 mispredict at mid-`L`. [`orion/src/algorithm/search/unified.rs:411–424`]
- **Flush cadence: `FLUSH_INTERVAL = [1, 4]`.** Pre-convergence flush every hop, post-convergence every 4 — amortises the merge fixed cost when admissions thin out. [`orion/src/algorithm/search/in_mem_search.rs:266–270`]
- **3-way merge routing** at flush time, picked by `K` (admits) vs `L` (PQ capacity):
  - `K · 8 < L` → per-element `pq.insert` (cache-friendly small-K, no scratch swap)
  - `L / 8 ≤ K ≤ L · 0.67` → `pq.batch_merge_gallop` (`partition_point` + bulk memcpy, +10–30 ns)
  - `K · 1.5 > L` → linear `pq.batch_merge` (two-way set-union, near-capacity admits)

  Comparisons lower to pure shifts (`K << 3`, `K + (K >> 1)`). [`orion/src/algorithm/search/in_mem_search.rs:97–110`]
- **Reversible convergence + early exit.** Single converged hop doesn't lock out navigation — protects against false-trigger recall loss. Once stably converged, `EarlyExitChecker` terminates on N consecutive zero-admission hops. GIST-960 L=200: cuts ~20 % of distance calls for < 0.4 pp recall loss. [`orion/src/algorithm/search/convergence.rs`, `early_exit.rs`]

#### 5. Rerank tier (`--rerank`)

Once the beam has converged (or early-exited), the top `k × RERANK_FACTOR = 20` PQ entries are re-scored with the highest-precision distance available. The rerank tier walks each survivor once — it never sees the rejection rate of the per-hop admission tier.

| Option | Cost / vert | Score | Used by |
|---|---|---|---|
| `f32` | 4 lines (D=128) | direct L2 | L2 cascades (SIFT, GIST, Deep10M, Fashion-MNIST) |
| `ip-f32` | 4 lines | raw IP | MIPS cascades (GloVe, MS-MARCO, Wiki-ada) |
| `u16` | 2 lines + i32 sidecar | precise quantized L2 | high-recall L2 variant |

- **Flat walk over a fixed 20-entry list.** The rerank stage reads the top `k × RERANK_FACTOR = 20` IDs straight off `scratch.pq` and streams them through `DistanceStream<L2F32Distance>` / `DistanceStream<IpF32Distance>` (`LA_TRUTH = 6`, env `ORION_DSTREAM_LA_TRUTH`). No graph traversal here — the beam loop's `rerank_candidates(id) = (local, extra)` two-slice walk is part of the **post-convergence expansion**, not this stage.
- **Register-renaming-friendly single-point kernels** are the only kernel optimisation in flight: the `*_neon_distance.rs` kernels' 4 independent FMA accumulator chains (`s0..s3`) sustain ~1 FMA / cycle even though the 20 vertices arrive cold off DRAM. Detail in [Shared kernels → distance functions](#shared-kernels).

#### 6. Shared kernels

Three pieces used by every stage above:

**`DistanceStream` — inline per-cache-line prefetch + flat 4-way resolve.** Drives every per-hop distance kernel (admission slab + rerank slab + JL prefilter slab). Resolves every prfm address inline from `(base_ptr, ids[v], stride, line_offset)`; the scheduler keeps `lookahead_lines` in flight without per-call queue setup. Layout `(lpv, stride)` is classified once into one of four address-arithmetic specialisations (`lpv=1`+pow2, `lpv` pow2 + stride pow2, `lpv=1`+non-pow2, div/mod fallback for GIST f32 lpv=30) — drops the per-iter `csel`-driven `lsl`/`mul` waste. `prfm pldl1strm` (= `_MM_HINT_NTA`) frees cache slots faster since lines are read at most `lpv` times per query and never reused across queries. Lookahead env-overridable via `ORION_DSTREAM_LA_Q` / `ORION_DSTREAM_LA_TRUTH` (defaults 10 / 6). [`vector/src/distance_stream.rs:114–540`, 4-way dispatch at `:503–540`]

**Distance functions — `vector/src/*_neon_distance.rs`.** The NEON kernel set the admission + rerank tiers parameterise over. **Every kernel is 4× internally unrolled into independent accumulator chains** so the CPU's register-renamer dispatches four FMAs / SDOTs in flight per chunk — single-accumulator forces RAW on one phys reg (1 FMA / ~4 cy on the M2 FMA pipe), 4-chain saturates the pipeline (~1 FMA / cy). Reductions fold the chains as `vaddq_f32(s0, s1) + vaddq_f32(s2, s3)` then `vaddvq_f32`. [single-point `vector/src/ip_neon_distance.rs:38–82`] A speculative **batch-4 candidate variant** (`distance_ip_vector_f32_batch4`, `:111–148`) applies the same trick *across* candidates so 4 cold-DRAM reads could overlap through the M2 MSHR queue, but it's not currently wired into any callsite — the cold-cache rerank top-20 walk goes through plain `DistanceStream<IpF32Distance>` with the single-point kernel, which `LA_TRUTH = 6` prefetch hides the misses for.

**`Neighbor` PQ — `total_cmp`.** Branchless integer-compare `Neighbor::cmp` — no `Option` overhead from default `f32::partial_cmp`. Combined with `pad16` alignment (above) this is what makes the 3-way merge cadence in §4(b) hold steady at low `L`. [`orion/src/model/neighbor/neighbor.rs:48`]

#### Per-dataset default cascade

| Dataset | Cascade | Rationale |
|---|---|---|
| SIFT, Deep10M | `none → l2-u8 → f32` | low/mid-D L2; direct u8 admission, no prefilter needed |
| GloVe-25, GloVe-100 | `none → mips-i8 → ip-f32` | normalised angular; single-phase i8 sdot |
| GIST | `jl → l2-kt → f32` | D=960 — JL prefilter at 1 line/vert gates the wider L2-kt slab; kernel-trick recovers exact L2 ranking via `sdot` |
| Fashion-MNIST | `none → l2-kt → f32` | 60 K × 784 — too small for JL setup tax; L2-kt admission is the speed knob |
| MS-MARCO BERT 1M | `none → mips-i8 → ip-f32` | raw MIPS; JL empirically hurts both QPS and recall on this dataset (setup tax + filters genuine top-K) |
| Wiki-ada-002 1M | `jl → mips-i8 → ip-f32` | high-D cosine; JL at 1 line/vert gates the 1536-D admission slab |

#### Cascade-wide invariants

What makes the cascade compose:

- **Metric-agnostic auto-calibration.** `calibrate()` derives `(threshold, early_exit_limit)` from admit-rate topology, not the distance function — same output drives every metric. [`orion/src/algorithm/search/calibrate.rs`]
- **No normalize at load.** Earlier MIPS paths L2-normalised at load, silently converting raw MIPS → cosine and breaking `msmarco_bert_1M` against its raw-dot-product GT. The whole `normalize_all` infrastructure was removed; admission tiers either consume unit-norm-by-construction input (glove, wiki-ada) or raw vectors (msmarco_bert).
- **PA 2-pass base graph default.** Every entry in `sweep.yaml` points at `${PA_ROOT}/data/<ds>/<stub>.staged` (PA `-num_passes 2` refine; ~10 % matched-recall lift over a 1-pass build; ~3 s one-time import). [`benchmark/src/runner/parlayann_bridge.rs`]
- **Cascade-uniform machinery.** Adding a new admission tier means writing one `QuantSpec` impl — the preparation, peeled warm-up, per-hop cadence, rerank, and shared kernels are reused unchanged.
