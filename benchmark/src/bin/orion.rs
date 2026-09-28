//! Orion sweep driver: resolve config, load data, dispatch a compiled dimension.

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

#[path = "../config.rs"]
mod config;

use clap::Parser;
#[path = "../cli/mod.rs"]
mod cli;
use cli::config::{Args, ResolvedRunConfig, VectorStorageKind};
use cli::data::LoadedDataset;
use orion::algorithm::search::calibrate::CalibrationElement;

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
        log::error!(
            "[mlock] {label}: failed to pin {} MiB ({err}) — falling back to pageable",
            len / (1024 * 1024)
        );
    } else {
        log::info!("[mlock] {label}: pinned {} MiB", len / (1024 * 1024));
    }
}

macro_rules! log_search_metrics {
    // Terminal
    (@parse
        [fmt: $($fmt:expr,)*]
        [args: $($args:tt)*]
    ) => {
        log::info!(
            concat!($($fmt,)*),
            $($args)*
        )
    };

    // Literal layout fragment
    (@parse
        [fmt: $($fmt:expr,)*]
        [args: $($args:tt)*]
        $literal:literal
        $($rest:tt)*
    ) => {
        log_search_metrics!(
            @parse
            [fmt: $($fmt,)* $literal,]
            [args: $($args)*]
            $($rest)*
        )
    };

    // Record
    (@parse
        [fmt: $($fmt:expr,)*]
        [args: $($args:tt)*]
        (
            $present:ident,
            $origin:ident,
            $format:literal
            $(, $transform:expr)?
        )
        $($rest:tt)*
    ) => {
        log_search_metrics!(
            @parse
            [
                fmt:
                $($fmt,)*
                "{",
                stringify!($present),
                ":",
                $format,
                "}",
            ]
            [
                args:
                $($args)*
                $present = log_search_metrics!(
                    @value $origin $(, $transform)?
                ),
            ]
            $($rest)*
        )
    };

    // Value without conversion
    (@value $origin:ident) => {
        $origin
    };

    // Value with conversion
    (@value $origin:ident, $transform:expr) => {
        ($transform)($origin)
    };

    // Public entry -- MUST come last
    ($($item:tt)+) => {
        log_search_metrics!(
            @parse
            [fmt:]
            [args:]
            $($item)+
        )
    };
}

/// Storage-specific hooks used by the common QPS-recall sweep.
///
/// The benchmark driver keeps timing, calibration, sweep control, and reporting
/// identical across resident vector representations. Implementations of this
/// trait provide only the operations that depend on the resident element type:
///
/// - loading the corresponding [`Orion`](orion::Orion) index,
/// - pinning(mlocking) storage-specific search stage sidecars,
/// - dispatching the batch-search implementation.
///
/// This keeps the benchmark procedure shared between `f32` and native `u8`
/// storage while allowing each representation to use its own index and search
/// pipeline.
trait SweepElement: cli::data::BaseElement + CalibrationElement {
    /// Loads or constructs an index whose resident base representation is `Self`.
    ///
    /// The implementation may select a storage-specific loading path, but ownership
    /// of the base allocation is transferred from `data` rather than copied.
    fn load<const N: usize>(
        config: &ResolvedRunConfig,
        data: &mut LoadedDataset<Self>,
    ) -> Result<orion::Orion<N, Self>, String>
    where
        [Self; N]: vector::FullPrecisionDistance<Self, N>;

    /// Pins any representation-specific search sidecar before the sweep begins.
    ///
    /// Implementations may use this hook to mlock cascade stage sidecars(auxiliary
    /// dataset used in these stages). It is intentionally allowed to be a no-op
    /// when the resident auxiliary sidecar has been pinned elsewhere(if it is the original
    /// dataset itself, it will be pinned with queries in main `run` function).
    fn pin<const N: usize>(index: &orion::Orion<N, Self>, config: &ResolvedRunConfig)
    where
        [Self; N]: vector::FullPrecisionDistance<Self, N>;

    /// Executes one batch-search point of the sweep.
    ///
    /// `beam` is the current search-list width selected by the sweep, while
    /// `calibration` contains the thresholds determined during calibration.
    ///
    /// The returned neighbor IDs are consumed by the common benchmark path for
    /// recall calculation and reporting.
    fn search<const N: usize>(
        index: &orion::Orion<N, Self>,
        queries: &[[f32; N]],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>>
    where
        [Self; N]: vector::FullPrecisionDistance<Self, N>;
}

/// Sweep implementation for the conventional `f32` resident representation.
///
/// This path uses the configured cascade directly, including any prefilter,
/// admission sidecar, and rerank stage selected by [`ResolvedRunConfig`].
impl SweepElement for f32 {
    fn load<const N: usize>(
        config: &ResolvedRunConfig,
        data: &mut LoadedDataset<Self>,
    ) -> Result<orion::Orion<N>, String> {
        cli::index::load_index(config, data)
    }

    fn pin<const N: usize>(index: &orion::Orion<N>, config: &ResolvedRunConfig) {
        cascade::pin_cascade(
            index,
            config.cascade.prefilter,
            config.cascade.admission,
            config.cascade.rerank,
        );
    }

    fn search<const N: usize>(
        index: &orion::Orion<N>,
        queries: &[[f32; N]],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>> {
        cascade::search_batch_compose(
            index,
            queries,
            config.sweep.k,
            beam,
            config.window_size,
            calibration.threshold,
            calibration.early_exit_limit,
            config.cascade.prefilter,
            config.cascade.admission,
            config.cascade.rerank,
        )
    }
}

/// Sweep implementation for native `u8` resident storage.
///
/// The base vectors remain in their original byte representation. Exact L2
/// admission is performed directly over the resident bytes, so this path does
/// not require a quantized admission sidecar or an `f32` rerank stage.
impl SweepElement for u8 {
    fn load<const N: usize>(
        config: &ResolvedRunConfig,
        data: &mut LoadedDataset<Self>,
    ) -> Result<orion::Orion<N, u8>, String> {
        cli::index::load_u8_index(config, data)
    }

    fn pin<const N: usize>(_index: &orion::Orion<N, u8>, _config: &ResolvedRunConfig) {
        // The common pin below covers the shared byte base; there is no sidecar.
    }

    fn search<const N: usize>(
        index: &orion::Orion<N, u8>,
        queries: &[[f32; N]],
        config: &ResolvedRunConfig,
        beam: usize,
        calibration: orion::CalibratedParams,
    ) -> diskann::common::ANNResult<Vec<Vec<u32>>> {
        use orion::algorithm::search::stage::{
            admission::NativeU8Admission, prefilter::NoPrefilter, rerank::NoRerank,
        };
        let admission = NativeU8Admission::new(&index.dataset);
        index.search_batch_unified(
            queries,
            config.sweep.k,
            beam,
            config.window_size,
            calibration.threshold,
            calibration.early_exit_limit,
            None::<&NoPrefilter>,
            &admission,
            &NoRerank,
        )
    }
}

fn run_sweep<const N: usize, T: SweepElement>(
    config: &ResolvedRunConfig,
    mut data: LoadedDataset<T>,
) -> Result<(), String>
where
    [T; N]: vector::FullPrecisionDistance<T, N>,
{
    let run = &config.sweep;
    let k = run.k;
    let num_threads = run.threads;
    let idx = T::load::<N>(config, &mut data)?;
    if config.prepare_only {
        return Ok(());
    }

    let gt = &data.ground_truth;
    let queries_arr: Vec<[f32; N]> = data
        .queries
        .iter()
        .map(|q| {
            let mut a = [0f32; N];
            a.copy_from_slice(&q[..N]);
            a
        })
        .collect();

    // Per-cascade sidecar materialisation. `pin_cascade` below
    // would also trigger these via `ensure_*`, but we kick them
    // here so the build/load latency is logged separately from
    // the timed sweep's setup.
    log::info!("Cascade: {}", config.cascade.label());

    // Calibrate on the full graph with the cascade's exact metric
    // and top-k target, including unnormalized MIPS datasets.
    let calib_qs: Vec<[f32; N]> =
        queries_arr[..run.calibration_samples.min(queries_arr.len())].to_vec();
    let calib = idx
        .calibrate(
            &calib_qs,
            run.calibration_l,
            config.window_size,
            orion::CalibrationConfig {
                k,
                metric: config.metric.calibration_metric(),
            },
        )
        .expect("calibrate");
    let threshold = calib.threshold;
    let early_exit_limit = calib.early_exit_limit;
    log::info!(
        "Calibrated ({}): threshold={:.2}, early_exit_limit={}",
        config.cascade.label(),
        threshold,
        early_exit_limit
    );

    // Select the k-specific YAML schedule unless the CLI overrides it.
    let ls: &[usize] = &run.search_list_sizes;
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
    let trials = run.trials;

    // Profile-ready signal: after all prep (load, normalize,
    // calibrate, quantize) is done but before the measured sweep
    // starts, drop a marker file so `profile_mips_q.sh` can attach
    // xctrace at the right moment and only capture the search
    // phase. Mirrors `search_profile.rs`'s pattern.
    if profiling {
        let pid = std::process::id();
        let marker = format!("/tmp/orion_sweep_{}.ready", pid);
        std::fs::write(&marker, pid.to_string()).expect("write profile marker");
        log::error!(
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
        log::error!("Starting profiled search section ({} trials).", trials);
    }

    // QoS hint applied to every rayon worker at thread creation —
    // this is what keeps the 8 search threads pinned to the 6 P-
    // cores + (boosted) 2 E-cores on M2 instead of getting
    // demoted to background QoS during DVFS thermal events. Also
    // bump the main thread (drives the `pool.install` block).
    set_thread_qos_user_interactive();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .start_handler(|_| set_thread_qos_user_interactive())
        .build()
        .unwrap();
    log::info!(
        "═══ Orion cascade (R={}, α={}, ws={}, cascade={}) ═══",
        config.graph_degree,
        config.alpha,
        config.window_size,
        config.cascade.label()
    );

    // Per-cascade sidecars + mlock pins. `pin_cascade` walks the
    // three axes (prefilter / admission / rerank), materialises
    // each tier's sidecar via the orion accessor, and pins the
    // byte ranges. Idempotent — safe to call before warmup and
    // before the timed sweep.
    T::pin(&idx, config);
    // Both storage paths pin their resident base, graph slab and query batch.
    // Native u8 owns only this one base buffer, with no f32 or quantized copy.
    {
        let ds_bytes_ptr = idx.dataset.data.as_ptr() as *const u8;
        let ds_bytes_len = idx.dataset.data.len() * std::mem::size_of::<T>();
        mlock_bytes("resident base", ds_bytes_ptr, ds_bytes_len);
        let pg = idx.graph.buffer_bytes();
        mlock_bytes("pgraph slab", pg.as_ptr(), pg.len());
        let q_bytes_ptr = queries_arr.as_ptr() as *const u8;
        let q_bytes_len = queries_arr.len() * std::mem::size_of::<[f32; N]>();
        mlock_bytes("queries", q_bytes_ptr, q_bytes_len);
    }

    // Warm up at the first measured L, which has already been checked >= k.
    let warmup_l = ls[0];
    // 1 warmup rounds: drives DVFS to peak P-state, primes
    // prefetchers, and stabilises rayon worker pool placement
    // before the timed sweep. A single warmup pass left the
    // first timed L (16) ~5–10% slower than its run-2 reading
    // due to lingering cold-cache + DVFS-ramp jitter.
    for _ in 0..1 {
        let _warm =
            pool.install(|| T::search(&idx, &queries_arr, config, warmup_l, calib).unwrap());
        drop(_warm);
    }
    // Reset the per-phase atomic counters so the warmup's stats
    // don't contaminate the first timed L's averages.
    {
        use orion::algorithm::search::{
            NDC_F32, NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS,
            QUERY_COUNT, RAW_VISIT_COUNT, SETUP_NS, VISIT_COUNT,
        };
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
            let results = pool.install(|| T::search(&idx, &queries_arr, config, l, calib).unwrap());
            let wall = t.elapsed();
            samples.push(queries_arr.len() as f64 / wall.as_secs_f64());
            recall = mean_recall(&results, gt, k);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let qps = samples[trials / 2];
        // Visit / per-phase instrumentation (mips_q path only):
        // drain global counters and print averages per query.
        // Compares directly against PA's `average visited` and
        // exposes pre/post-convergence yield for diagnosis.

        use orion::algorithm::search::{
            NDC_F32, NDC_I8, POST_CONV_ADMITS, POST_CONV_HOPS, PRE_CONV_ADMITS, PRE_CONV_HOPS,
            QUERY_COUNT, RAW_VISIT_COUNT, SETUP_NS, VISIT_COUNT,
        };
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
            let yield_pre = if pre_h > 0 {
                pre_a as f64 / pre_h as f64
            } else {
                0.0
            };
            let yield_post = if post_h > 0 {
                post_a as f64 / post_h as f64
            } else {
                0.0
            };
            let avg_ndc_i8 = ndc_i8 as f64 / q as f64;
            let avg_ndc_f32 = ndc_f32 as f64 / q as f64;
            let avg_ndc_total = avg_ndc_i8 + avg_ndc_f32;
            let avg_setup_us = setup_ns as f64 / q as f64 / 1000.0;
            // Total wall-clock time spent in per-thread setup,
            // summed across all threads. Trial wall ≈ N / QPS;
            // total CPU-time across threads ≈ trial_wall × num_threads.
            // setup_total / cpu_total tells us the absolute fraction
            // of CPU spent on setup work.
            let setup_total_s = setup_ns as f64 / 1e9;
            let trial_wall_s = q as f64 / qps;
            let cpu_total_s = trial_wall_s * num_threads as f64;
            let setup_pct = if cpu_total_s > 0.0 {
                100.0 * setup_total_s / cpu_total_s
            } else {
                0.0
            };

            log_search_metrics!(
                " L="           (l,             l,             ">4")
                "  R@"          (k,             k,             "")
                "="             (recall,        recall,        ".4")
                "  QPS="        (qps,           qps,           ".0")

                "  visits="     (visits,        avg_v,         ".0")
                "  raw="        (raw,           avg_raw,       ".0")
                "  filter="     (filter_pct,    filter_ratio,  ".1", |x| x * 100.0)
                "%"

                "  cmps="       (cmps,          avg_ndc_total, ".0")
                " (i8="         (i8,            avg_ndc_i8,    ".0")
                " + f32="       (f32,           avg_ndc_f32,   ".0")
                ")"

                " setup="       (setup,         avg_setup_us,  ".1")
                "µs (total="    (setup_total,   setup_total_s, ".3")
                "s, "           (setup_pct,     setup_pct,     ".1")
                "%)"

                " hops_pre="    (hops_pre,      avg_pre,       ".1")
                " (a/h="        (yield_pre,     yield_pre,     ".1")
                ")"

                " hops_post="   (hops_post,    avg_post,       ".1")
                " (a/h="        (yield_post,   yield_post,     ".1")
                ")"
            );
        } else {
            log_search_metrics!(
                " L="           (l,             l,             ">4")
                "  R@"          (k,             k,             "")
                "="             (recall,        recall,        ".4")
                "  QPS="        (qps,           qps,           ".0")
            );
        }
    }

    Ok(())
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

fn execute(args: Args) -> Result<(), String> {
    let config = args.resolve()?;
    let resolved = serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?;
    if args.print_config {
        log::info!("{resolved}");
        return Ok(());
    }
    log::info!("Run settings: {resolved}");
    if args.preflight {
        println!(
            "{}",
            serde_json::to_string_pretty(&cli::resources::preflight_report(&config)?)
                .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    match config.vector_storage {
        VectorStorageKind::F32 => {
            let data = LoadedDataset::<f32>::load(&config)?;
            with_supported_dimension!(config.dimension, |D| run_sweep::<D, f32>(&config, data))
        }
        VectorStorageKind::U8 => {
            let data = LoadedDataset::<u8>::load(&config)?;
            // Resolution restricts native byte mode to SIFT's physical dimension.
            run_sweep::<128, u8>(&config, data)
        }
    }
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    if let Err(error) = execute(Args::parse()) {
        log::error!("{error}");
        std::process::exit(2);
    }
}
