/// Staged-only sweep — loads (or builds) a PhasedGraph cache and runs the
/// QPS-recall sweep on StagedDiskANN alone. No DiskANN / DiskANN-matched
/// baselines.
///
/// ## Datasets
///   sift      — SIFT1M (128-dim, L2)
///   glove25   — GloVe-25 (32-dim, pre-normalized, angular)
///   glove100  — GloVe-100 (100-dim, pre-normalized, angular)
///   gist      — GIST-1M (960-dim, L2)
///
/// `glove25` / `glove100` always point at pre-normalized fvecs under
/// `data/glove*_norm/` and build with R=100 L=200 α=1.0 — matching
/// ParlayANN's `vamana/scripts/{glove25,glove100}`. The raw-data
/// variants were retired since on angular data they are strictly
/// dominated.
///
/// ## Metrics (`--metric`)
///   * `l2`     — 2-phase u8 prefilter + f32 L2 rerank (non-angular path)
///   * `mips`   — single-phase `-⟨q, v⟩` on unit-normalized data
///   * `mips-q` — PA-style i8 beam + f32 top-`k × 2` rerank (angular
///                high-dim; wins on glove100 by ~10% over PA native)
///
/// When `--metric` is omitted, per-dataset defaults kick in:
///   glove100 → mips-q,  everything else → l2.
///
/// ## CLI flags
///   <dataset>                     positional, required
///   --metric l2|l2-q|mips|mips-q       override per-dataset default
///   --max-points N                truncate base set
///   --search-list-sizes L1,L2,... comma-separated custom L schedule —
///                                 replaces the built-in 14-value sweep
///                                 (useful for profiling with a fixed
///                                 workload, e.g. `--search-list-sizes 16`)
///
/// ## Env vars
///   STAGED_GRAPH=pa            load PhasedGraph from the PA-import
///                              cache (`cache/staged_parlayann/`) instead
///                              of our own rust-built cache
///   STAGED_STAGED_FILE=<path>  path to a ParlayANN `.staged v2` export;
///                              if set and the PA-import cache is missing,
///                              staged_sweep imports the `.staged` on the
///                              fly (via `parlayann_bridge`) and saves
///                              the PhasedGraph to cache
///
/// ## Examples
///   ./target/release/staged_sweep glove100                            # MIPS-Q by default
///   ./target/release/staged_sweep glove100 --metric mips
///   ./target/release/staged_sweep sift --max-points 100000
///   STAGED_GRAPH=pa ./target/release/staged_sweep glove100
///   ./target/release/staged_sweep glove100 --search-list-sizes 16      # single L (profile)
///   ./target/release/staged_sweep glove100 --search-list-sizes 16,64,256
use std::time::Instant;


// Pull parlayann_bridge in directly — it's a standalone module (only
// std + rayon deps) so `#[path]` import keeps it accessible from this
// bin without touching the crate's `runner` module graph.
#[path = "../runner/parlayann_bridge.rs"]
mod parlayann_bridge;

#[path = "../utils.rs"]
mod utils;

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
/// builds). Tracks `defaults.staged.max_extra` in
/// `benchmark/configs/sweep.yaml`. 16 controls PhasedGraph slot stride
/// directly — bigger caps balloon stride and torpedo cache locality.
const DEFAULT_MAX_EXTRA: usize = 16;
/// Convergence checker's sliding window size. Tracks
/// `defaults.staged.window_size` in the yaml.
const DEFAULT_WINDOW_SIZE: usize = 5;

#[derive(Copy, Clone, Eq, PartialEq)]
enum Metric {
    L2,
    /// L2 with **u8 quantized beam + f32 L2 rerank** (mirror of
    /// `MipsQ` for the SIFT family). PQ holds u8 quantized distances
    /// throughout, with the L2 path's 3-hop loop peel preserved at
    /// the start; final stage reranks the top `k · RERANK_FACTOR` via
    /// f32 truth. Targets the bandwidth-saved-vs-noise-recovered
    /// trade-off — single-stage u8 hop is ~4× cheaper per cmp than
    /// the L2 stage-1+stage-2 shape, with rerank closing the recall
    /// gap. Sidecar `.qds`.
    L2Q,
    Mips,
    /// MIPS with i8 quantized beam + f32 IP rerank (two-phase, PA-style
    /// `-quantize_bits 8`). Only meaningful on unit-normalized data;
    /// targets high-dim angular (GloVe-100+) where single-phase f32 IP
    /// is memory-bandwidth bound.
    MipsQ,
    /// Same architecture as `MipsQ` but **i16** beam storage —
    /// matches PA's `-quantize_bits 16 -quantize_mode 1` recipe. 256×
    /// more representable distance values lifts recall above the i8
    /// ceiling at high L (sidecar `.qdm16`, twice the bandwidth).
    MipsQI16,
}

macro_rules! run_sweep {
    ($dim_name:expr, $data:ident, $queries:ident, $n:ident, $N:literal, $alpha:expr,
     $r:expr, $build_l:expr, $max_extra:expr, $ws:expr, $gt_path:expr, $metric:expr,
     $search_list_sizes:expr) => {{
        use staged_diskann::{build_diskann_index, StagedDiskANN};

        // Under MIPS-family metrics we normalize the base in RAM before
        // build, so the resulting graph is distinct from the L2 graph on
        // the same dataset. Suffix the cache key with `_norm` so they
        // don't collide — **except** for datasets whose fvecs on disk
        // are already pre-normalized (both `glove25` and `glove100` —
        // the raw-data variants were retired), where the in-RAM
        // normalize is a no-op and the graph matches the L2-on-normalized
        // path directly, so no suffix is needed.
        let is_mips_family = matches!(
            $metric,
            Metric::Mips | Metric::MipsQ | Metric::MipsQI16
        );
        // L2-Q reuses the L2 graph + L2 quantized sidecar (`.qds`) — no
        // cache-key suffix needed; it's purely a search-time variant.
        let is_pre_normalized_on_disk =
            $dim_name == "glove25" || $dim_name == "glove100";
        let cache_key_owned: String = if is_mips_family && !is_pre_normalized_on_disk {
            format!("{}_norm", $dim_name)
        } else {
            $dim_name.to_string()
        };
        let cache_key: &str = &cache_key_owned;
        let alpha_tag = format!("{:.2}", $alpha).replace('.', "_");
        let use_pa_graph = std::env::var("STAGED_GRAPH")
            .map(|v| v == "pa")
            .unwrap_or(false);
        let (cache_dir, cache_path) = if use_pa_graph {
            // Cache stub mirrors the .staged filename's `_ex{N}_pct{P}`
            // suffix so each (max_extra, local_pct) combo gets its
            // own .pgraph cache. `STAGED_LOCAL_PCT` env defaults to 60
            // (matches the C++ default in `neighborsTime.C`).
            let local_pct: usize = std::env::var("STAGED_LOCAL_PCT")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|&v| (1..=100).contains(&v))
                .unwrap_or(60);
            let stub = format!("{}_ex{}_pct{}", cache_key, $max_extra, local_pct);
            let dir = std::path::PathBuf::from("cache/staged_parlayann");
            let p = dir.join(format!(
                "{}_{}_n{}_r{}_l{}_a{}_ex{}_pct{}.bin",
                cache_key, stub, $n, $r, $build_l, alpha_tag, $max_extra, local_pct
            ));
            (dir, p)
        } else {
            let dir = std::path::PathBuf::from("cache/staged");
            let p = dir.join(format!(
                "{}_n{}_r{}_l{}_a{}_ex{}.bin",
                cache_key, $n, $r, $build_l, alpha_tag, $max_extra
            ));
            (dir, p)
        };
        std::fs::create_dir_all(&cache_dir).ok();
        let pgraph_path = cache_path.with_extension("pgraph");

        let staged = if cache_path.exists() && pgraph_path.exists() {
            println!("Loading cached PhasedGraph from {:?}", pgraph_path);
            let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
            let mut idx = StagedDiskANN::<$N>::load_from_cache(&cache_path, empty_ds)
                .expect("load_from_cache failed");
            let mut ds = diskann::model::InmemDataset::<f32, $N>::new($n, 1.0).unwrap();
            ds.data.memcpy(&$data[..$n * $N]).unwrap();
            idx.dataset = ds;
            idx
        } else if !use_pa_graph {
            println!(
                "Building StagedDiskANN ({}-dim, R={}, L={}, α={}, ex={}) — will save to {:?}",
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
            let mut idx = StagedDiskANN::<$N>::new(
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
            // given by `STAGED_STAGED_FILE`; we save to the same cache
            // slot on the way out so the second invocation hits the fast
            // load-from-cache arm above.
            let staged_file_path = std::env::var("STAGED_STAGED_FILE")
                .expect(
                    "STAGED_GRAPH=pa with no cached pgraph — set \
                     STAGED_STAGED_FILE=<path.staged> so staged_sweep can import \
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
            let mut idx = StagedDiskANN::<$N>::new(
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

        // Per-metric quantized sidecar (L2 → .qds, MipsQ → .qdm8,
        // MipsQI16 → .qdm16, Mips → none).
        match $metric {
            Metric::L2 => {
                let _ = staged.ensure_quantized_dataset();
                println!("L2 mode — u8 prefilter + f32 L2 rerank");
            }
            Metric::L2Q => {
                let _ = staged.ensure_quantized_dataset();
                println!("L2-Q mode — single-stage u8 L2 beam + f32 top-20 rerank");
            }
            Metric::Mips => {
                println!("MIPS mode — single-phase f32 IP");
            }
            Metric::MipsQ => {
                let _ = staged.ensure_quantized_dataset_mips();
                println!("MIPS-Q mode — PA-style i8 beam + f32 top-20 rerank");
            }
            Metric::MipsQI16 => {
                let _ = staged.ensure_quantized_dataset_mips_i16();
                println!("MIPS-Q-i16 mode — PA-style i16 beam + f32 top-20 rerank");
            }
        }

        // Metric-agnostic calibration — derives `threshold` (dcc
        // convergence ε) + `early_exit_limit` from graph topology
        // alone (admit-rate inflection + P90 gap from last useful
        // admission). On unit-normalized angular data L2 and neg-IP
        // rank identically, so the calibrator's internal `Metric::L2`
        // compare still produces valid params for MIPS / MIPS-Q.
        let calib_qs: Vec<[f32; $N]> =
            queries_arr[..CALIB_SAMPLE.min(queries_arr.len())].to_vec();
        let calib = staged
            .calibrate(&calib_qs, CALIB_L, $ws)
            .expect("calibrate");
        let threshold = calib.threshold;
        let early_exit_limit = calib.early_exit_limit;
        println!(
            "Calibrated ({}): threshold={:.2}, early_exit_limit={}",
            match $metric {
                Metric::L2 => "L2",
                Metric::L2Q => "L2-Q",
                Metric::Mips => "MIPS",
                Metric::MipsQ => "MIPS-Q",
                Metric::MipsQI16 => "MIPS-Q-i16",
            },
            threshold, early_exit_limit
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
        let profiling = std::env::var("STAGED_PROFILE_MARKER").is_ok();
        let trials = if profiling { 30 } else { TRIALS };
        let k = K;

        // Profile-ready signal: after all prep (load, normalize,
        // calibrate, quantize) is done but before the measured sweep
        // starts, drop a marker file so `profile_mips_q.sh` can attach
        // xctrace at the right moment and only capture the search
        // phase. Mirrors `search_profile.rs`'s pattern.
        if profiling {
            let pid = std::process::id();
            let marker = format!("/tmp/staged_sweep_{}.ready", pid);
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
            "\n═══ Staged sweep (R={}, α={}, ws={}, metric={:?}) ═══",
            $r, $alpha, $ws,
            match $metric {
                Metric::L2 => "L2",
                Metric::L2Q => "L2-Q",
                Metric::Mips => "MIPS",
                Metric::MipsQ => "MIPS-Q",
                Metric::MipsQI16 => "MIPS-Q-i16",
            }
        );
        let gt = load_ivecs($gt_path).0;

        // Resolve the matching quantized base dataset *once*, before
        // entering the timed loop, so the lazy `OnceLock` build /
        // sidecar load cost is paid up-front and excluded from QPS.
        // Both refs hand-off to every search call below; the matching
        // turbofish (`::<i8>` / `::<i16>`) selects the right path.
        // L2 + L2-Q both use the u8 quantized sidecar (`.qds`).
        let q_ds_l2 = if matches!($metric, Metric::L2 | Metric::L2Q) {
            Some(staged.ensure_quantized_dataset())
        } else {
            None
        };
        // MipsQ uses i8 quantized dataset; pure Mips runs single-stage
        // f32 directly off `dataset` (no quantized sidecar).
        let q_ds_i8 = if matches!($metric, Metric::MipsQ) {
            Some(staged.ensure_quantized_dataset_mips())
        } else {
            None
        };
        let q_ds_i16 = if matches!($metric, Metric::MipsQI16) {
            Some(staged.ensure_quantized_dataset_mips_i16())
        } else {
            None
        };

        // Pin the three hot regions (f32 base dataset, quantized base
        // sidecar, PhasedGraph slab) into RAM via `mlock(2)` so they
        // cannot be paged out between the warmup pass and the timed
        // sweep. Without this, on a heavily-multitasked machine the
        // first few timed trials at a fresh L pay page-in latency that
        // skews the early QPS readings. Best-effort: pin failures fall
        // back to ordinary paging (logged, no abort).
        {
            let ds_bytes_ptr = staged.dataset.data.as_ptr() as *const u8;
            let ds_bytes_len = staged.dataset.data.len() * std::mem::size_of::<f32>();
            mlock_bytes("dataset (f32)", ds_bytes_ptr, ds_bytes_len);
            if let Some(q) = q_ds_l2 {
                mlock_bytes(
                    "qdataset (u8)",
                    q.data.as_ptr() as *const u8,
                    q.data.len() * std::mem::size_of::<u8>(),
                );
            }
            if let Some(q) = q_ds_i8 {
                mlock_bytes(
                    "qdataset (i8)",
                    q.data.as_ptr() as *const u8,
                    q.data.len() * std::mem::size_of::<i8>(),
                );
            }
            if let Some(q) = q_ds_i16 {
                mlock_bytes(
                    "qdataset (i16)",
                    q.data.as_ptr() as *const u8,
                    q.data.len() * std::mem::size_of::<i16>(),
                );
            }
            let pg = staged.graph.buffer_bytes();
            mlock_bytes("pgraph slab", pg.as_ptr(), pg.len());
            // Pin the query batch too — `flush_cache()` between trials
            // evicts L1/L2/SLC, so on a memory-pressured machine the
            // query pages can also drift out of resident set between
            // warmup and timed runs.
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
            let _warm = pool.install(|| match $metric {
                // L2 (SIFT family): u8 prefilter + per-hop f32 rerank.
                Metric::L2 => staged
                    .search_batch_l2_u8(
                        &queries_arr,
                        k,
                        warmup_l,
                        $ws,
                        threshold,
                        early_exit_limit,
                    )
                    .unwrap(),
                // L2-Q: single-stage u8 beam + post-hoc f32 rerank.
                Metric::L2Q => staged
                    .search_batch_l2_u8_q(
                        &queries_arr,
                        k,
                        warmup_l,
                        $ws,
                        threshold,
                        early_exit_limit,
                    )
                    .unwrap(),
                // MIPS pure (glove-25-style low-dim): single-stage f32, no quant.
                Metric::Mips => staged
                    .search_batch_mips(
                        &queries_arr,
                        k,
                        warmup_l,
                        $ws,
                        threshold,
                        early_exit_limit,
                    )
                    .unwrap(),
                // MIPS-Q (glove-100 i8): single-stage quantized + post-hoc rerank.
                Metric::MipsQ => staged
                    .search_batch_mips_q::<staged_diskann::model::MipsI8>(
                        &queries_arr,
                        q_ds_i8.unwrap(),
                        k,
                        warmup_l,
                        $ws,
                        threshold,
                        early_exit_limit,
                    )
                    .unwrap(),
                // MIPS-Q-i16: same as above but i16 storage.
                Metric::MipsQI16 => staged
                    .search_batch_mips_q::<staged_diskann::model::MipsI16>(
                        &queries_arr,
                        q_ds_i16.unwrap(),
                        k,
                        warmup_l,
                        $ws,
                        threshold,
                        early_exit_limit,
                    )
                    .unwrap(),
            });
            drop(_warm);
        }
        // Reset the per-phase atomic counters so the warmup's stats
        // don't contaminate the first timed L's averages.
        {
            use std::sync::atomic::Ordering;
            use staged_diskann::algorithm::search::in_mem_search_mips_q::*;
            VISIT_COUNT.store(0, Ordering::Relaxed);
            QUERY_COUNT.store(0, Ordering::Relaxed);
            PRE_CONV_HOPS.store(0, Ordering::Relaxed);
            POST_CONV_HOPS.store(0, Ordering::Relaxed);
            PRE_CONV_ADMITS.store(0, Ordering::Relaxed);
            POST_CONV_ADMITS.store(0, Ordering::Relaxed);
            NDC_I8.store(0, Ordering::Relaxed);
            NDC_F32.store(0, Ordering::Relaxed);
            SETUP_NS.store(0, Ordering::Relaxed);
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
                let results = pool.install(|| match $metric {
                    Metric::L2 => staged
                        .search_batch_l2_u8(
                            &queries_arr,
                            k,
                            l,
                            $ws,
                            threshold,
                            early_exit_limit,
                        )
                        .unwrap(),
                    Metric::L2Q => staged
                        .search_batch_l2_u8_q(
                            &queries_arr,
                            k,
                            l,
                            $ws,
                            threshold,
                            early_exit_limit,
                        )
                        .unwrap(),
                    Metric::Mips => staged
                        .search_batch_mips(
                            &queries_arr,
                            k,
                            l,
                            $ws,
                            threshold,
                            early_exit_limit,
                        )
                        .unwrap(),
                    Metric::MipsQ => staged
                        .search_batch_mips_q::<staged_diskann::model::MipsI8>(
                            &queries_arr,
                            q_ds_i8.unwrap(),
                            k,
                            l,
                            $ws,
                            threshold,
                            early_exit_limit,
                        )
                        .unwrap(),
                    Metric::MipsQI16 => staged
                        .search_batch_mips_q::<staged_diskann::model::MipsI16>(
                            &queries_arr,
                            q_ds_i16.unwrap(),
                            k,
                            l,
                            $ws,
                            threshold,
                            early_exit_limit,
                        )
                        .unwrap(),
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
            use std::sync::atomic::Ordering;
            use staged_diskann::algorithm::search::in_mem_search::*;
            let v = VISIT_COUNT.swap(0, Ordering::Relaxed);
            let q = QUERY_COUNT.swap(0, Ordering::Relaxed);
            let pre_h = PRE_CONV_HOPS.swap(0, Ordering::Relaxed);
            let post_h = POST_CONV_HOPS.swap(0, Ordering::Relaxed);
            let pre_a = PRE_CONV_ADMITS.swap(0, Ordering::Relaxed);
            let post_a = POST_CONV_ADMITS.swap(0, Ordering::Relaxed);
            let ndc_i8 = NDC_I8.swap(0, Ordering::Relaxed);
            let ndc_f32 = NDC_F32.swap(0, Ordering::Relaxed);
            let setup_ns = SETUP_NS.swap(0, Ordering::Relaxed);
            if q > 0 {
                let avg_v = v as f64 / q as f64;
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
                    "  L={l:>4}  R@10={recall:.4}  QPS={qps:.0}  visits={avg_v:.0}  cmps={avg_ndc_total:.0} (i8={avg_ndc_i8:.0}+f32={avg_ndc_f32:.0})  setup={avg_setup_us:.1}µs (total={setup_total_s:.3}s, {setup_pct:.1}%)  hops_pre={avg_pre:.1} (a/h={yield_pre:.1})  hops_post={avg_post:.1} (a/h={yield_post:.1})"
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

/// L2-normalize every `dim`-stride slice of `data` in place.
fn normalize_all(data: &mut [f32], dim: usize) {
    for chunk in data.chunks_exact_mut(dim) {
        vector::l2_normalize_f32_inplace(chunk);
    }
}

/// Minimal CLI parser: first positional = dataset, then flags.
///   --metric l2|mips
///   --max-points N
fn parse_args() -> (String, usize, Metric, Vec<usize>) {
    let args: Vec<String> = std::env::args().collect();
    let mut dataset = "sift".to_string();
    let mut max_points: usize = usize::MAX;
    let mut metric: Option<Metric> = None;
    let mut search_list_sizes: Option<Vec<usize>> = None;
    let mut i = 1;
    let mut seen_positional = false;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--metric" => {
                i += 1;
                metric = Some(match args.get(i).map(|s| s.as_str()) {
                    Some("l2") => Metric::L2,
                    Some("l2-q") | Some("l2-q-u8") => Metric::L2Q,
                    Some("mips") => Metric::Mips,
                    Some("mips-q") | Some("mips-q-i8") => Metric::MipsQ,
                    Some("mips-q-i16") => Metric::MipsQI16,
                    Some(v) => {
                        panic!("Unknown metric: {v} (expected l2|l2-q|mips|mips-q|mips-q-i16)")
                    }
                    None => panic!("--metric requires an argument"),
                });
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
    // Per-dataset default metric when `--metric` is omitted. `glove100`
    // is a 100-dim angular dataset where MIPS-Q (PA-style i8 beam +
    // f32 rerank) is the production choice — it beats plain L2 by 2×+
    // and we want a bare `staged_sweep glove100` to reflect that.
    // Other datasets fall through to L2 (historical default).
    let metric = metric.unwrap_or_else(|| match dataset.as_str() {
        "sift" => Metric::L2Q,
        "glove25" => Metric::Mips,
        "glove100" => Metric::MipsQ,
        _ => Metric::L2,
    });

    // Default L schedule = the 14-value curve used for QPS-recall plots.
    // `--search-list-sizes` overrides it (e.g. a single L for profiling).
    let search_list_sizes = search_list_sizes.unwrap_or_else(|| DEFAULT_L_SCHEDULE.to_vec());
    (dataset, max_points, metric, search_list_sizes)
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    let (dataset, max_points, metric, search_list_sizes) = parse_args();

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
            "gist",
            "data/gist/gist_base.fvecs",
            "data/gist/gist_query.fvecs",
            "data/gist/gist_groundtruth.ivecs",
            960usize,
            1.5f32,
            32usize,
            48usize,
        ),
        _ => panic!("Unknown dataset: {dataset}"),
    };

    println!(
        "Loading {dataset} (dim={dim}, metric={}, max_points={})...",
        match metric {
            Metric::L2 => "L2",
            Metric::L2Q => "L2-Q",
            Metric::Mips => "MIPS",
            Metric::MipsQ => "MIPS-Q",
            Metric::MipsQI16 => "MIPS-Q-i16",
        },
        max_points
    );
    let (mut data, n, _) = load_fvecs(base_path, max_points);
    let (mut qdata, nq, _) = load_fvecs(query_path, usize::MAX);
    // Normalize both base and query under any MIPS-family metric. We still
    // accept raw fvecs on disk — the L2 path is untouched.
    if matches!(metric, Metric::Mips | Metric::MipsQ | Metric::MipsQI16) {
        normalize_all(&mut data, dim);
        normalize_all(&mut qdata, dim);
        println!("Normalized {n} base + {nq} query vectors to unit length");
    }
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
            metric,
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
            metric,
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
            metric,
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
            metric,
            search_list_sizes
        ),
        _ => panic!("Unsupported dimension: {dim}"),
    }
}
