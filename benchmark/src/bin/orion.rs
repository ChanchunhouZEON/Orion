/// Orion sweep — production benchmark driver. Search dispatch
/// goes through the composable cascade (`--prefilter`, `--admission`,
/// `--rerank`); per-dataset defaults pick the production cascade
/// triple when the flags are omitted.
///
/// ## Datasets
///   sift      — SIFT1M (128-dim, L2)
///   glove25   — GloVe-25 (32-dim, pre-normalized, angular)
///   glove100  — GloVe-100 (100-dim, pre-normalized, angular)
///   gist      — GIST-1M (960-dim, L2)
///
/// ## Cascade axes
///   --prefilter <none|jl>
///     none — graph neighbours go straight to admission (right for
///            low-D where the admission tier is already cheap).
///     jl   — 1024-bit JL Sparse Hamming filter (high-D L2).
///
///   --admission <l2-u8|l2-u16|l2-kt|mips-i8|mips-i16>
///     l2-u8     — direct u8 L2, 1 cache line / 64 dims.
///     l2-u16    — u16 L2 (PA's `-quantize_bits 16`).
///     l2-kt     — i8 sdot + per-vert `‖x‖²` reconstruction.
///     mips-i8   — i8 MIPS (sdot on unit-normalised data).
///     mips-i16  — i16 MIPS for the high-recall band.
///
///   --rerank <f32|ip-f32|u16>
///     f32     — f32 base, L2 rerank (default for L2 cascades).
///     ip-f32  — f32 base, IP rerank — for MIPS cascades; unit-
///               normalises the query inside the rerank.
///     u16     — u16 sidecar, L2 rerank (PA-bit-exact recipe).
///
/// Per-dataset defaults pick a sensible triple when the flags are
/// omitted — see `Cascade::default_for_dataset`.
///
/// ## CLI flags
///   <dataset>                     positional, required
///   --prefilter X --admission Y --rerank Z   cascade triple
///   --max-points N                truncate base set
///   --search-list-sizes L1,L2,... comma-separated custom L schedule
///
/// ## Env vars (shared with orion)
///   ORION_GRAPH=pa            use the PA-import cache.
///   ORION_STAGED_FILE=<path>   PA `.staged v2` export path.
///
/// ## Examples
///   ./target/release/orion glove100
///   ./target/release/orion glove100 --admission mips-i16 --rerank ip-f32
///   ./target/release/orion gist --prefilter jl --admission l2-u16
use std::time::Instant;

// Pull parlayann_bridge in directly — it's a standalone module (only
// std + rayon deps) so `#[path]` import keeps it accessible from this
// bin without touching the crate's `runner` module graph.
#[path = "../runner/parlayann_bridge.rs"]
mod parlayann_bridge;

#[path = "../utils.rs"]
mod utils;

// Cascade dispatch lives in the runner module (next to the runners
// that consume the same orion-dispatch primitives). Imported via
// `#[path]` because the bin is not a benchmark-lib consumer.
#[path = "../runner/cascade.rs"]
mod cascade;

use cascade::{AdmissionChoice, PrefilterChoice, RerankChoice};

/// Promote the calling thread to the highest QoS tier (`USER_INTERACTIVE`,
/// 0x21) so macOS keeps it on a P-core and at the highest DVFS step
/// instead of demoting it to E-cores under thermal pressure. Set on
/// every rayon worker (via `start_handler`) and on the main thread
/// before the timed sweep — the mid-band 2-3× QPS jitter we see on
/// glove100 traces back to threads getting transiently parked on
/// E-cores (1/3 P-core perf), which this hint suppresses.
///
/// Apple QoS classes (per `<pthread/qos.h>` / Foundation docs):
///   USER_INTERACTIVE = 0x21  (highest — UI responsiveness)
///   USER_INITIATED   = 0x19
///   DEFAULT          = 0x15
///   UTILITY          = 0x11
///   BACKGROUND       = 0x09  (lowest — definitely E-cores)
#[cfg(target_os = "macos")]
fn set_thread_qos_user_interactive() {
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    const QOS_CLASS_USER_INTERACTIVE: u32 = 0x21;
    // Best-effort: ignore the return code — there's no fallback that
    // would help us, and the call is advisory anyway.
    unsafe {
        let _ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}
#[cfg(not(target_os = "macos"))]
fn set_thread_qos_user_interactive() {}

/// `mlock(2)` a byte range so its pages cannot be paged out. We pin
/// dataset + quantized base + PhasedGraph slab right after construction
/// so the warmup pass and the timed sweep both see exactly the same
/// resident pages — no eviction in between, no page-in cost on the
/// timed-side critical path. Best-effort: a non-zero return is logged
/// but does not abort the sweep (e.g. `RLIMIT_MEMLOCK` exceeded just
/// means we fall back to ordinary paging — same behavior as before).
fn mlock_bytes(label: &str, ptr: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    let rc = unsafe { libc::mlock(ptr as *const libc::c_void, len) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        eprintln!(
            "[mlock] {label}: failed to pin {} MiB ({err}) — falling back to pageable",
            len / (1024 * 1024)
        );
    } else {
        println!("[mlock] {label}: pinned {} MiB", len / (1024 * 1024));
    }
}

// ── Sweep-wide tuning constants ───────────────────────────────────────────
/// Top-`k` retrieved per query — fixed across every benchmark for
/// apples-to-apples comparison with ParlayANN's `10@10 recall` reports.
const K: usize = 10;
/// Number of timed repeats per `L` value. Median is taken as the per-L
/// QPS reading. With `flush_cache()` called between trials (ParlayANN
/// `check_nn_recall.h`-style 40 MB eviction), every trial starts from
/// a cold cache — so a single trial per `L` already matches the
/// "first query" conditions PA measures on their end. Historical
/// higher values (7, 11) pre-date the cache-flush and were there to
/// average out hot-cache noise.
const TRIALS: usize = 1;
/// Warm-up queries fed into `calibrate()` to derive `threshold` and
/// `early_exit_limit`. 200 converges well before diminishing returns
/// on all four reference datasets.
const CALIB_SAMPLE: usize = 200;
/// `search_list_size` used **inside** the calibration search (not the
/// outer sweep). Mid-range value — calibrator wants a full beam to
/// observe convergence dynamics but not so large the warmup dominates
/// the build time.
const CALIB_L: usize = 48;
/// Rayon pool size for parallel query execution. Matches the 8-core
/// M2 Pro topology we've been tuning against. Could be made env-
/// overridable later if we ever move to a different CPU.
const NUM_THREADS: usize = 8;
/// Default L schedule for the QPS-recall sweep. Densely sampled (~28
/// values, ratio < 1.25× between neighbors) so the per-L work grows
/// smoothly across the sweep — consecutive measurements stay near the
/// previous one's working set, the OS scheduler doesn't migrate
/// threads, and DVFS sees a steady ramp instead of step-changes that
/// trigger frequency oscillation. Mirrors PA's `{10, 11, 12, ..., 30,
/// 32, 34, ...}` density. Individual runs can override via
/// `--search-list-sizes L1,L2,...` (e.g. a single 16 for profile runs).
const DEFAULT_L_SCHEDULE: &[usize] = &[
    16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 90, 100, 114, 128, 144,
    160, 180, 200, 224, 256, 288, 320, 384, 448, 512, 640, 768, 1024,
];
/// Max extras per vertex — hard cap on the per-node `extras` zone after
/// the top-X% partition (PA's `-max_extra` flag, mirrored in rust-native
/// builds). Tracks `defaults.orion.max_extra` in
/// `benchmark/configs/sweep.yaml`. 16 controls PhasedGraph slot stride
/// directly — bigger caps balloon stride and torpedo cache locality.
const DEFAULT_MAX_EXTRA: usize = 16;
/// Convergence checker's sliding window size. Tracks
/// `defaults.orion.window_size` in the yaml.
const DEFAULT_WINDOW_SIZE: usize = 5;

/// CLI-driven cascade composition. Three orthogonal axes (prefilter,
/// admission, rerank) the caller selects independently via
/// `--prefilter/--admission/--rerank`. Per-dataset defaults pick a
/// reasonable triple when the flags are omitted — see
/// [`Cascade::default_for_dataset`].
///
/// The triple drives every downstream decision in `run_sweep!`:
///   * Cache-key naming (MIPS-family triples get the `_norm` suffix
///     unless the source fvecs are already unit-normalised).
///   * Base-vector normalisation pass before graph build.
///   * Which sidecars [`cascade::pin_cascade`] materialises + mlocks.
///   * Which `(P, A, R)` Box<dyn> trio
///     [`cascade::search_batch_compose`] hands to
///     `search_batch_unified`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Cascade {
    prefilter: PrefilterChoice,
    admission: AdmissionChoice,
    rerank: RerankChoice,
}

impl Cascade {
    /// Default cascade per dataset when the flags are omitted.
    ///   * SIFT / deep10m (low-D L2): direct u8 admission, no prefilter
    ///   * glove25 / glove100 (low-D MIPS): i8 sdot admission
    ///   * GIST (D=960 L2): JL prefilter + kernel-trick admission (high-D L2)
    ///   * msmarco_bert_1M / wiki_ada_1M (high-D MIPS): JL + i8 MIPS
    ///   * unknown: low-D L2 baseline
    fn default_for_dataset(dataset: &str) -> Self {
        match dataset {
            // deep10m: Yandex Deep1B's first 10M CNN image features.
            // Native D=96, padded to D=128 (32-byte SIMD chunk
            // alignment). PA's recipe is `-dist_func Euclidian` so
            // this is the **L2** family, not angular — same cascade
            // as SIFT (direct u8 admission).
            "sift" | "deep10m" => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::L2U8,
                rerank: RerankChoice::F32,
            },
            // fashion-mnist (60K × 784, L2): 60K is too small for the JL
            // prefilter's per-query setup overhead to amortise, but
            // D=784 is wide enough that the L2-Kt kernel-trick (per-vert
            // ‖x‖² precompute → single dot product per compare instead
            // of L2's two subtractions per dim) wins by ~30% vs naive
            // L2-U8. Net: no prefilter, l2-kt admission, f32 rerank.
            "fashion-mnist" => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::L2Kt,
                rerank: RerankChoice::F32,
            },
            "glove25" | "glove100" => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::MipsI8,
                rerank: RerankChoice::IpF32,
            },
            // GIST (D=960 L2): JL prefilter at 1 line/vert gates the
            // wider admission read — pays for itself once the
            // admission slab crosses ~8 cache lines / vert.
            "gist" => Self {
                prefilter: PrefilterChoice::Jl,
                admission: AdmissionChoice::L2Kt,
                rerank: RerankChoice::F32,
            },
            // msmarco_bert_1M: 1M MS-MARCO passages embedded with
            // sentence-transformers/msmarco-bert-base-dot-v5 (BERT-base,
            // 768-D, trained for dot product). Empirical sweep shows JL
            // prefilter actively hurts on this dataset on BOTH axes:
            // the ~9µs setup tax dominates at high L (QPS −60% at
            // L=1024) and the filter drops enough genuine top-K
            // candidates to nudge recall slightly down. No prefilter +
            // MIPS-i8 admission + f32-IP rerank is strictly better.
            "msmarco_bert_1M" => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::MipsI8,
                rerank: RerankChoice::IpF32,
            },
            // wiki_ada_1M: 1M Wikipedia passages × 1536-D OpenAI ada-002
            // embeddings, sourced from nlpkevinl/wikipedia_openai_embeddings.
            // ada-002 outputs are L2-unit-normalised by construction
            // (see OpenAI's `https://platform.openai.com/docs/guides/embeddings`),
            // so dot-product and cosine rank identically — MIPS family
            // applies directly without re-normalising at load. Same
            // (jl, mips-i8, ip-f32) shape as msmarco_bert_1M, just twice
            // as wide on the admission slab (96 lines/vert vs 48).
            "wiki_ada_1M" => Self {
                prefilter: PrefilterChoice::Jl,
                admission: AdmissionChoice::MipsI8,
                rerank: RerankChoice::IpF32,
            },
            _ => Self {
                prefilter: PrefilterChoice::None,
                admission: AdmissionChoice::L2U8,
                rerank: RerankChoice::F32,
            },
        }
    }

    /// Short human-readable label, e.g. `jl→l2-kt→f32` — used in the
    /// banner print.
    fn label(&self) -> String {
        let p = match self.prefilter {
            PrefilterChoice::None => "none",
            PrefilterChoice::Jl => "jl",
            PrefilterChoice::JlHadamard => "jl-hadamard",
            PrefilterChoice::Rabitq => "rabitq",
        };
        let a = match self.admission {
            AdmissionChoice::L2U8 => "l2-u8",
            AdmissionChoice::L2U16 => "l2-u16",
            AdmissionChoice::L2Kt => "l2-kt",
            AdmissionChoice::MipsI8 => "mips-i8",
            AdmissionChoice::MipsI16 => "mips-i16",
            AdmissionChoice::AdsF32 => "ads-f32",
        };
        let r = match self.rerank {
            RerankChoice::F32 => "f32",
            RerankChoice::IpF32 => "ip-f32",
            RerankChoice::U16 => "u16",
            RerankChoice::None => "none",
        };
        format!("{p}→{a}→{r}")
    }
}

macro_rules! run_sweep {
    ($dim_name:expr, $data:ident, $queries:ident, $n:ident, $N:literal, $alpha:expr,
     $r:expr, $build_l:expr, $max_extra:expr, $ws:expr, $gt_path:expr, $cascade:expr,
     $search_list_sizes:expr) => {{
        use orion::{build_diskann_index, Orion};

        let cache_key: &str = $dim_name;
        let alpha_tag = format!("{:.2}", $alpha).replace('.', "_");
        let use_pa_graph = std::env::var("ORION_GRAPH")
            .map(|v| v == "pa")
            .unwrap_or(false);
        let (cache_dir, cache_path) = if use_pa_graph {
            // Append `_pct{P}` so each `(max_extra, local_pct)` combo
            // gets its own .pgraph cache. `ORION_LOCAL_PCT` env
            // defaults to 60 (matches the C++ default in
            // `neighborsTime.C`).
            let local_pct: usize = std::env::var("ORION_LOCAL_PCT")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&v| (1..=100).contains(&v))
                .unwrap_or(60);
            let dir = std::path::PathBuf::from("cache/orion_parlayann");
            let p = dir.join(format!(
                "{}_n{}_r{}_l{}_a{}_ex{}_pct{}.bin",
                cache_key, $n, $r, $build_l, alpha_tag, $max_extra, local_pct
            ));
            (dir, p)
        } else {
            let dir = std::path::PathBuf::from("cache/orion");
            let p = dir.join(format!(
                "{}_n{}_r{}_l{}_a{}_ex{}.bin",
                cache_key, $n, $r, $build_l, alpha_tag, $max_extra
            ));
            (dir, p)
        };
        std::fs::create_dir_all(&cache_dir).ok();
        let pgraph_path = cache_path.with_extension("pgraph");

        let idx = if cache_path.exists() && pgraph_path.exists() {
            println!("Loading cached PhasedGraph from {:?}", pgraph_path);
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = Orion::<$N>::load_from_cache(&cache_path, empty_ds)
                .expect("load_from_cache failed");
            let mut ds = diskann::model::InmemDataset::<f32, $N>::new($n, 1.0).unwrap();
            ds.data.memcpy(&$data[..$n * $N]).unwrap();
            idx.dataset = ds;
            idx
        } else if !use_pa_graph {
            println!(
                "Building Orion ({}-dim, R={}, L={}, α={}, ex={}) — will save to {:?}",
                $N, $r, $build_l, $alpha, $max_extra, pgraph_path
            );
            let t0 = Instant::now();
            let result = build_diskann_index(
                &$data, $n, $N, $alpha as f32, $r as u32, $build_l as u32,
                false, None, None, true, $max_extra,
            )
            .expect("build_diskann_index failed");
            let entry = result.entry_point;
            drop(result.index);
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = Orion::<$N>::new(
                empty_ds, &result.partitions, entry,
                $r, $max_extra, None, None, Some(cache_path.clone()), true,
            );
            let mut ds = diskann::model::InmemDataset::<f32, $N>::new($n, 1.0).unwrap();
            ds.data.memcpy(&$data[..$n * $N]).unwrap();
            idx.dataset = ds;
            println!("Build+save took {:.1}s", t0.elapsed().as_secs_f64());
            idx
        } else {
            // PA-graph cache is missing and build_diskann_index isn't our
            // path — instead import the `.staged` export PA wrote. Path
            // given by `ORION_STAGED_FILE`; we save to the same cache
            // slot on the way out so the second invocation hits the fast
            // load-from-cache arm above.
            let staged_file_path = std::env::var("ORION_STAGED_FILE")
                .expect(
                    "ORION_GRAPH=pa with no cached pgraph — set \
                     ORION_STAGED_FILE=<path.staged> so orion can import \
                     the ParlayANN-built graph (produced by \
                     `prepare_parlayann_data.sh`)",
                );
            println!(
                "Importing ParlayANN .staged export from {} — will save \
                 PhasedGraph cache to {:?}",
                staged_file_path, pgraph_path
            );
            let t0 = Instant::now();
            let input = parlayann_bridge::load_from_staged_file(
                &staged_file_path,
                &$data[..$n * $N],
                $N,
            )
            .expect("parlayann_bridge::load_from_staged_file failed");
            let entry = input.entry_point;
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            // Use the file-header's `max_extra` (= max per-node extras
            // count after the top-X% partition), not the dataset
            // config's static value. The partition rule produces a
            // variable per-node extras count up to this max.
            let max_extra_from_file = input.max_extra as usize;
            let mut idx = Orion::<$N>::new(
                empty_ds, &input.partitions, entry,
                $r, max_extra_from_file, None, None, Some(cache_path.clone()), true,
            );
            let mut ds = diskann::model::InmemDataset::<f32, $N>::new($n, 1.0).unwrap();
            ds.data.memcpy(&$data[..$n * $N]).unwrap();
            idx.dataset = ds;
            println!(
                "Import+save took {:.1}s (n={}, max_deg={}, max_extra={})",
                t0.elapsed().as_secs_f64(),
                input.num_nodes,
                input.max_deg,
                input.max_extra
            );
            idx
        };

        let queries_arr: Vec<[f32; $N]> = $queries
            .iter()
            .map(|q| {
                let mut a = [0f32; $N];
                a.copy_from_slice(&q[..$N]);
                a
            })
            .collect();

        // Per-cascade sidecar materialisation. `pin_cascade` below
        // would also trigger these via `ensure_*`, but we kick them
        // here so the build/load latency is logged separately from
        // the timed sweep's setup.
        println!("Cascade: {}", $cascade.label());

        // Cascade-agnostic calibration — derives `threshold` (dcc
        // convergence ε) + `early_exit_limit` from graph topology
        // alone (admit-rate inflection + P90 gap from last useful
        // admission). On unit-normalized angular data L2 and neg-IP
        // rank identically, so the calibrator's internal L2 compare
        // produces valid params for MIPS cascades too.
        let calib_qs: Vec<[f32; $N]> =
            queries_arr[..CALIB_SAMPLE.min(queries_arr.len())].to_vec();
        let calib = idx
            .calibrate(&calib_qs, CALIB_L, $ws)
            .expect("calibrate");
        let threshold = calib.threshold;
        let early_exit_limit = calib.early_exit_limit;
        println!(
            "Calibrated ({}): threshold={:.2}, early_exit_limit={}",
            $cascade.label(),
            threshold,
            early_exit_limit
        );

        // L schedule comes from CLI (`--search-list-sizes`), default =
        // the standard 14-value curve, but can be overridden (e.g. a
        // single `16` for profiling a fixed workload).
        let ls: &[usize] = &$search_list_sizes;
        // Under `STAGED_PROFILE_MARKER`, repeat the timed sweep enough
        // times that xctrace's CPU-Profile sampler (1 ms / sample, ~8
        // threads) has multiple seconds of actual search work to
        // catch. Empty traces from earlier runs were caused by the
        // attached profiler missing the entire sweep at small L
        // (single-trial L=16 finishes in ~100 ms — barely 100 sample
        // slots before the process exits and Instruments writes
        // mostly-empty `1.run` buffers). 30 trials × default L
        // schedule keeps the timed region > 10 s.
        let profiling = std::env::var("ORION_PROFILE_MARKER").is_ok();
        let trials = if profiling { 30 } else { TRIALS };
        let k = K;

        // Profile-ready signal: after all prep (load, normalize,
        // calibrate, quantize) is done but before the measured sweep
        // starts, drop a marker file so `profile_mips_q.sh` can attach
        // xctrace at the right moment and only capture the search
        // phase. Mirrors `search_profile.rs`'s pattern.
        if profiling {
            let pid = std::process::id();
            let marker = format!("/tmp/orion_sweep_{}.ready", pid);
            std::fs::write(&marker, pid.to_string())
                .expect("write profile marker");
            eprintln!(
                "PID={pid} — marker {marker} written. Sleeping 15s so xctrace finishes attach + buffer init before the sweep starts..."
            );
            // 15 s window: xctrace's `record --attach` needs 2-4 s to
            // load the CPU-Profile template, allocate ring buffers,
            // hook the kdebug stream, and start sampling on M2. The
            // earlier 5 s was on the edge — most of it got eaten by
            // setup, leaving sub-second of actual sampling that
            // happened to overlap the prep tail rather than the
            // measured sweep. 15 s gives a comfortable margin.
            std::thread::sleep(std::time::Duration::from_secs(15));
            let _ = std::fs::remove_file(&marker);
            eprintln!("Starting profiled search section ({} trials).", trials);
        }

        // QoS hint applied to every rayon worker at thread creation —
        // this is what keeps the 8 search threads pinned to the 6 P-
        // cores + (boosted) 2 E-cores on M2 instead of getting
        // demoted to background QoS during DVFS thermal events. Also
        // bump the main thread (drives the `pool.install` block).
        set_thread_qos_user_interactive();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .start_handler(|_| set_thread_qos_user_interactive())
            .build()
            .unwrap();
        println!(
            "\n═══ Orion cascade (R={}, α={}, ws={}, cascade={}) ═══",
            $r, $alpha, $ws, $cascade.label()
        );
        let gt = load_ivecs($gt_path).0;

        // Per-cascade sidecars + mlock pins. `pin_cascade` walks the
        // three axes (prefilter / admission / rerank), materialises
        // each tier's sidecar via the orion accessor, and pins the
        // byte ranges. Idempotent — safe to call before warmup and
        // before the timed sweep.
        cascade::pin_cascade(
            &idx,
            $cascade.prefilter,
            $cascade.admission,
            $cascade.rerank,
        );
        // Pin the f32 base + PhasedGraph slab + query batch, which
        // are cascade-agnostic. (The cascade's rerank stage already
        // pins the f32 base for `F32` / `IpF32`, but pinning here
        // covers `U16` rerank too.)
        {
            let ds_bytes_ptr = idx.dataset.data.as_ptr() as *const u8;
            let ds_bytes_len = idx.dataset.data.len() * std::mem::size_of::<f32>();
            mlock_bytes("dataset (f32)", ds_bytes_ptr, ds_bytes_len);
            let pg = idx.graph.buffer_bytes();
            mlock_bytes("pgraph slab", pg.as_ptr(), pg.len());
            let q_bytes_ptr = queries_arr.as_ptr() as *const u8;
            let q_bytes_len = queries_arr.len() * std::mem::size_of::<[f32; $N]>();
            mlock_bytes("queries", q_bytes_ptr, q_bytes_len);
        }

        // Warmup: a single full pass at L=1024 — a deep beam touches
        // the broadest portion of the graph + vector base, exercising
        // every page the timed sweep will visit. With the hot regions
        // mlocked above, the page state survives until the timed run,
        // so a single warmup pass suffices to prime prefetchers, DVFS,
        // and the rayon worker pool. No sleep between warmup and timed
        // sweep — the timed-sweep's per-trial `flush_cache()` re-cools
        // L1/L2/SLC anyway, and pinned pages survive any DRAM eviction
        // pressure that might otherwise build during a sleep window.
        let warmup_l = 16usize;
        // 1 warmup rounds: drives DVFS to peak P-state, primes
        // prefetchers, and stabilises rayon worker pool placement
        // before the timed sweep. A single warmup pass left the
        // first timed L (16) ~5–10% slower than its run-2 reading
        // due to lingering cold-cache + DVFS-ramp jitter.
        for _ in 0..1 {
            let _warm = pool.install(|| {
                cascade::search_batch_compose(
                    &idx,
                    &queries_arr,
                    k,
                    warmup_l,
                    $ws,
                    threshold,
                    early_exit_limit,
                    $cascade.prefilter,
                    $cascade.admission,
                    $cascade.rerank,
                )
                .unwrap()
            });
            drop(_warm);
        }
        // Reset the per-phase atomic counters so the warmup's stats
        // don't contaminate the first timed L's averages.
        {

            use orion::algorithm::search::{NDC_F32, NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, QUERY_COUNT, RAW_VISIT_COUNT, SETUP_NS, VISIT_COUNT};
            VISIT_COUNT.reset();
            RAW_VISIT_COUNT.reset();
            QUERY_COUNT.reset();
            PRE_CONV_HOPS.reset();
            POST_CONV_HOPS.reset();
            PRE_CONV_ADMITS.reset();
            POST_CONV_ADMITS.reset();
            NDC_I8.reset();
            NDC_F32.reset();
            SETUP_NS.reset();
        }

        for &l in ls {
            let mut samples = Vec::with_capacity(trials);
            let mut recall = 0.0f64;
            for _ in 0..trials {
                // PA-style 40 MB cache flush before each timed region — every
                // trial observes cold L1/L2/SLC so graph + vector loads match
                // the "first query" conditions PA reports.
                utils::flush_cache();
                let t = Instant::now();
                let results = pool.install(|| {
                    cascade::search_batch_compose(
                        &idx,
                        &queries_arr,
                        k,
                        l,
                        $ws,
                        threshold,
                        early_exit_limit,
                        $cascade.prefilter,
                        $cascade.admission,
                        $cascade.rerank,
                    )
                    .unwrap()
                });
                let wall = t.elapsed();
                samples.push(queries_arr.len() as f64 / wall.as_secs_f64());
                recall = mean_recall(&results, &gt, k);
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let qps = samples[trials / 2];
            // Visit / per-phase instrumentation (mips_q path only):
            // drain global counters and print averages per query.
            // Compares directly against PA's `average visited` and
            // exposes pre/post-convergence yield for diagnosis.

            use orion::algorithm::search::{NDC_F32, NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS, QUERY_COUNT, RAW_VISIT_COUNT, SETUP_NS, VISIT_COUNT};
            let v = VISIT_COUNT.drain();
            let raw_v = RAW_VISIT_COUNT.drain();
            let q = QUERY_COUNT.drain();
            let pre_h = PRE_CONV_HOPS.drain();
            let post_h = POST_CONV_HOPS.drain();
            let pre_a = PRE_CONV_ADMITS.drain();
            let post_a = POST_CONV_ADMITS.drain();
            let ndc_i8 = NDC_I8.drain();
            let ndc_f32 = NDC_F32.drain();
            let setup_ns = SETUP_NS.drain();
            if q > 0 {
                let avg_v = v as f64 / q as f64;
                let avg_raw = raw_v as f64 / q as f64;
                // Filter ratio: fraction of pre-admission candidates
                // the prefilter rejected. 0.0 when no prefilter ran.
                let filter_ratio = if raw_v > v {
                    (raw_v - v) as f64 / raw_v as f64
                } else {
                    0.0
                };
                let avg_pre = pre_h as f64 / q as f64;
                let avg_post = post_h as f64 / q as f64;
                let yield_pre = if pre_h > 0 { pre_a as f64 / pre_h as f64 } else { 0.0 };
                let yield_post = if post_h > 0 { post_a as f64 / post_h as f64 } else { 0.0 };
                let avg_ndc_i8 = ndc_i8 as f64 / q as f64;
                let avg_ndc_f32 = ndc_f32 as f64 / q as f64;
                let avg_ndc_total = avg_ndc_i8 + avg_ndc_f32;
                let avg_setup_us = setup_ns as f64 / q as f64 / 1000.0;
                // Total wall-clock time spent in per-thread setup,
                // summed across all threads. Trial wall ≈ N / QPS;
                // total CPU-time across threads ≈ trial_wall × NUM_THREADS.
                // setup_total / cpu_total tells us the absolute fraction
                // of CPU spent on setup work.
                let setup_total_s = setup_ns as f64 / 1e9;
                let trial_wall_s = q as f64 / qps;
                let cpu_total_s = trial_wall_s * NUM_THREADS as f64;
                let setup_pct = if cpu_total_s > 0.0 {
                    100.0 * setup_total_s / cpu_total_s
                } else {
                    0.0
                };
                println!(
                    "  L={l:>4}  R@10={recall:.4}  QPS={qps:.0}  visits={avg_v:.0}  raw={avg_raw:.0}  filter={filter_pct:.1}%  cmps={avg_ndc_total:.0} (i8={avg_ndc_i8:.0}+f32={avg_ndc_f32:.0})  setup={avg_setup_us:.1}µs (total={setup_total_s:.3}s, {setup_pct:.1}%)  hops_pre={avg_pre:.1} (a/h={yield_pre:.1})  hops_post={avg_post:.1} (a/h={yield_post:.1})",
                    filter_pct = filter_ratio * 100.0,
                );
            } else {
                println!("  L={l:>4}  R@10={recall:.4}  QPS={qps:.0}");
            }
        }
    }};
}

fn load_fvecs(path: &str, max_points: usize) -> (Vec<f32>, usize, usize) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("Cannot open {path}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let dim = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let record_bytes = 4 + dim * 4;
    let total = buf.len() / record_bytes;
    let n = total.min(max_points);
    let mut data = Vec::with_capacity(n * dim);
    for i in 0..n {
        let base = i * record_bytes + 4;
        for d in 0..dim {
            let off = base + d * 4;
            data.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
    }
    (data, n, dim)
}

fn load_ivecs(path: &str) -> (Vec<Vec<u32>>, usize) {
    use std::io::Read;
    let mut f = std::fs::File::open(path).unwrap_or_else(|_| panic!("Cannot open {path}"));
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).unwrap();
    let k = u32::from_le_bytes(buf[0..4].try_into().unwrap()) as usize;
    let record_bytes = 4 + k * 4;
    let n = buf.len() / record_bytes;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * record_bytes + 4;
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            let off = base + j * 4;
            row.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
        }
        out.push(row);
    }
    (out, n)
}

fn mean_recall(results: &[Vec<u32>], gt: &[Vec<u32>], k: usize) -> f64 {
    use std::collections::HashSet;
    let mut sum = 0.0f64;
    let n = results.len();
    for i in 0..n {
        let truth: HashSet<u32> = gt[i].iter().take(k).copied().collect();
        let hits = results[i].iter().filter(|x| truth.contains(x)).count();
        sum += hits as f64 / k as f64;
    }
    sum / n as f64
}

/// Minimal CLI parser: first positional = dataset, then flags.
///   --prefilter <none|jl>
///   --admission <l2-u8|l2-u16|l2-kt|mips-i8|mips-i16>
///   --rerank <f32|ip-f32|u16>
///   --max-points N
///   --search-list-sizes L1,L2,...
fn parse_args() -> (String, usize, Cascade, Vec<usize>) {
    use std::str::FromStr;
    let args: Vec<String> = std::env::args().collect();
    let mut dataset = "sift".to_string();
    let mut max_points: usize = usize::MAX;
    let mut prefilter_override: Option<PrefilterChoice> = None;
    let mut admission_override: Option<AdmissionChoice> = None;
    let mut rerank_override: Option<RerankChoice> = None;
    let mut search_list_sizes: Option<Vec<usize>> = None;
    let mut i = 1;
    let mut seen_positional = false;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--prefilter" => {
                i += 1;
                let v = args
                    .get(i)
                    .expect("--prefilter requires an argument (none|jl)");
                prefilter_override = Some(
                    PrefilterChoice::from_str(v).unwrap_or_else(|e| panic!("--prefilter: {e}")),
                );
            }
            "--admission" => {
                i += 1;
                let v = args.get(i).expect("--admission requires an argument");
                admission_override = Some(
                    AdmissionChoice::from_str(v).unwrap_or_else(|e| panic!("--admission: {e}")),
                );
            }
            "--rerank" => {
                i += 1;
                let v = args.get(i).expect("--rerank requires an argument");
                rerank_override =
                    Some(RerankChoice::from_str(v).unwrap_or_else(|e| panic!("--rerank: {e}")));
            }
            "--max-points" => {
                i += 1;
                max_points = args
                    .get(i)
                    .and_then(|s| s.parse().ok())
                    .expect("--max-points requires a positive integer");
            }
            "--search-list-sizes" | "--ls" => {
                i += 1;
                let raw = args
                    .get(i)
                    .expect("--search-list-sizes requires a comma-separated list, e.g. 16,32,64");
                let parsed: Vec<usize> = raw
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| {
                        s.parse::<usize>()
                            .unwrap_or_else(|_| panic!("bad L value in --search-list-sizes: {s:?}"))
                    })
                    .collect();
                assert!(!parsed.is_empty(), "--search-list-sizes list is empty");
                search_list_sizes = Some(parsed);
            }
            _ if !seen_positional => {
                dataset = a.clone();
                seen_positional = true;
            }
            _ => {
                // Back-compat: second positional = max_points (old form).
                if let Ok(v) = a.parse::<usize>() {
                    max_points = v;
                } else {
                    panic!("Unknown argument: {a}");
                }
            }
        }
        i += 1;
    }

    // Start from the per-dataset default; let each per-axis flag
    // override the corresponding field.
    let mut cascade = Cascade::default_for_dataset(&dataset);
    if let Some(p) = prefilter_override {
        cascade.prefilter = p;
    }
    if let Some(a) = admission_override {
        cascade.admission = a;
    }
    if let Some(r) = rerank_override {
        cascade.rerank = r;
    }

    let search_list_sizes = search_list_sizes.unwrap_or_else(|| DEFAULT_L_SCHEDULE.to_vec());
    (dataset, max_points, cascade, search_list_sizes)
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    let (dataset, max_points, cascade, search_list_sizes) = parse_args();

    // (cache_key, base_path, query_path, gt_path, dim, alpha, R, L)
    let (cache_key, base_path, query_path, gt_path, dim, alpha, r, build_l) = match dataset.as_str()
    {
        "sift" => (
            "sift",
            "data/sift/sift_base.fvecs",
            "data/sift/sift_query.fvecs",
            "data/sift/sift_groundtruth.ivecs",
            128usize,
            1.15f32,
            64usize,
            128usize,
        ),
        // GloVe-25 / GloVe-100 are **always** the PA-aligned recipe: they
        // read pre-normalized fvecs and build with R=100 L=200 α=1.0,
        // matching ParlayANN's `vamana/scripts/{glove25,glove100}`. The
        // raw-data variants were retired — on angular data they are
        // strictly dominated by the aligned path.
        "glove25" => (
            "glove25",
            "data/glove25_norm/glove-25-angular_base.fvecs",
            "data/glove25_norm/glove-25-angular_query.fvecs",
            "data/glove25_norm/glove-25-angular_groundtruth.ivecs",
            32usize,
            1.0f32,
            100usize,
            200usize,
        ),
        "glove100" => (
            "glove100",
            "data/glove100_norm/glove-100-angular_base.fvecs",
            "data/glove100_norm/glove-100-angular_query.fvecs",
            "data/glove100_norm/glove-100-angular_groundtruth.ivecs",
            100usize,
            1.0f32,
            100usize,
            200usize,
        ),
        "gist" => (
            // Aligned with `../ParlayANN/algorithms/vamana/scripts/gist`:
            //   BUILD_ARGS="-R 100 -L 200 -alpha 1.1 -num_passes 2"
            // Replaces the prior local recipe (R=32 L=48 α=1.5) which
            // built a much sparser graph and capped recall on the high
            // band. New params produce a denser graph at higher build
            // cost (~15-30 min on 1M D=960) but match the public
            // ParlayANN reference for apples-to-apples comparison.
            "gist",
            "data/gist/gist_base.fvecs",
            "data/gist/gist_query.fvecs",
            "data/gist/gist_groundtruth.ivecs",
            960usize,
            1.1f32,
            100usize,
            200usize,
        ),
        // Deep10M — Yandex Deep1B's first 10M CNN features (96-dim,
        // L2 ground truth). Mirrors `../ParlayANN/algorithms/vamana/
        // scripts/deep10M`: R=64 L=128 α=1.05 num_passes=2 -dist_func
        // Euclidian. Native D=96 → zero-padded to D=128 by
        // `convert_fbin.py --pad-to-dim 128`. The 128-dim mono-
        // morphisation has clean 32-byte SIMD chunk alignment
        // (128 / 32 = 4 chunks exactly), whereas D=100 would hit
        // `chunks_per_vert = 3.125` and lose the trailing 32-bit
        // SIMD lane in the inner loop. Zero padding contributes 0
        // to L2 — bit-identical ranking to native D=96.
        "deep10m" => (
            "deep10m",
            "data/deep10m/deep10m_base.fvecs",
            "data/deep10m/deep10m_query.fvecs",
            "data/deep10m/deep10m_groundtruth.ivecs",
            128usize,
            1.05f32,
            64usize,
            128usize,
        ),
        // Fashion-MNIST — 60K × 784-D image vectors (28×28 flattened
        // pixel intensities), L2 ground truth. Mirrors PA's
        // `vamana/scripts/fashion`: R=40 L=80 α=1.1 num_passes=2
        // -quantize_bits 8 -dist_func Euclidian. Files were generated
        // upstream as `fashion-mnist-784-euclidean_*` (ann-benchmarks
        // naming convention) so the path keeps that prefix.
        "fashion-mnist" => (
            "fashion-mnist",
            "data/fashion-mnist/fashion-mnist-784-euclidean_base.fvecs",
            "data/fashion-mnist/fashion-mnist-784-euclidean_query.fvecs",
            "data/fashion-mnist/fashion-mnist-784-euclidean_groundtruth.ivecs",
            784usize,
            1.1f32,
            40usize,
            80usize,
        ),
        // msmarco_bert_1M — 1M MS-MARCO passages embedded with
        // sentence-transformers/msmarco-bert-base-dot-v5 (BERT-base,
        // 768-D, dot-product). Groundtruth is brute-force top-100
        // MIPS against the *un-normalised* embedding (MPS matmul);
        // the runner L2-normalises both base + query at load so the
        // MIPS-family admission/rerank tiers receive unit-norm input
        // (cosine ranking — a faithful approximation of the model's
        // dot-product objective at unit-ball scale). D=768 = 24 × 32,
        // chunk-aligned. R=64 / L=128 mirrors PA's
        // `vamana/scripts/msmarco_websearch` build recipe.
        "msmarco_bert_1M" => (
            "msmarco_bert_1M",
            "data/msmarco_bert_1M/msmarco_bert_1M_base.fvecs",
            "data/msmarco_bert_1M/msmarco_bert_1M_query.fvecs",
            "data/msmarco_bert_1M/msmarco_bert_1M_groundtruth.ivecs",
            768usize,
            1.0f32,
            64usize,
            128usize,
        ),
        // wiki_ada_1M — 1M Wikipedia passages × 1536-D OpenAI ada-002
        // embeddings, sourced from nlpkevinl/wikipedia_openai_embeddings.
        // Queries are held out from the same corpus (ann-benchmarks
        // self-query convention — no separate question set exists for
        // this dump). Groundtruth is exact MIPS top-100 on MPS. Build
        // recipe: high-D MIPS shape (R=100 L=200 α=1.05).
        "wiki_ada_1M" => (
            "wiki_ada_1M",
            "data/wiki_ada_1M/wiki_ada_1M_base.fvecs",
            "data/wiki_ada_1M/wiki_ada_1M_query.fvecs",
            "data/wiki_ada_1M/wiki_ada_1M_groundtruth.ivecs",
            1536usize,
            1.05f32,
            100usize,
            200usize,
        ),
        _ => panic!("Unknown dataset: {dataset}"),
    };

    println!(
        "Loading {dataset} (dim={dim}, cascade={}, max_points={})...",
        cascade.label(),
        max_points
    );
    let (data, n, _) = load_fvecs(base_path, max_points);
    let (qdata, nq, _) = load_fvecs(query_path, usize::MAX);
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|i| qdata[i * dim..(i + 1) * dim].to_vec())
        .collect();
    println!("Loaded {n} base, {nq} queries.");

    let max_extra = DEFAULT_MAX_EXTRA;
    let ws = DEFAULT_WINDOW_SIZE;

    match dim {
        25 | 32 => run_sweep!(
            cache_key,
            data,
            queries,
            n,
            32,
            alpha,
            r as u32,
            build_l,
            max_extra,
            ws,
            gt_path,
            cascade,
            search_list_sizes
        ),
        100 => run_sweep!(
            cache_key,
            data,
            queries,
            n,
            100,
            alpha,
            r as u32,
            build_l,
            max_extra,
            ws,
            gt_path,
            cascade,
            search_list_sizes
        ),
        128 => run_sweep!(
            cache_key,
            data,
            queries,
            n,
            128,
            alpha,
            r as u32,
            build_l,
            max_extra,
            ws,
            gt_path,
            cascade,
            search_list_sizes
        ),
        768 => run_sweep!(
            cache_key,
            data,
            queries,
            n,
            768,
            alpha,
            r as u32,
            build_l,
            max_extra,
            ws,
            gt_path,
            cascade,
            search_list_sizes
        ),
        784 => run_sweep!(
            cache_key,
            data,
            queries,
            n,
            784,
            alpha,
            r as u32,
            build_l,
            max_extra,
            ws,
            gt_path,
            cascade,
            search_list_sizes
        ),
        960 => run_sweep!(
            cache_key,
            data,
            queries,
            n,
            960,
            alpha,
            r as u32,
            build_l,
            max_extra,
            ws,
            gt_path,
            cascade,
            search_list_sizes
        ),
        1536 => run_sweep!(
            cache_key,
            data,
            queries,
            n,
            1536,
            alpha,
            r as u32,
            build_l,
            max_extra,
            ws,
            gt_path,
            cascade,
            search_list_sizes
        ),
        _ => panic!("Unsupported dimension: {dim}"),
    }
}
