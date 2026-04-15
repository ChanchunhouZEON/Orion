/// Minimal search-only binary for cache profiling with xctrace CPU Counters.
///
/// Usage:
///   cargo build --release --bin search_profile
///   xctrace record --template "CPU Counters" --output /tmp/sift.trace \
///     -- ./target/release/search_profile sift 100000
///   xctrace record --template "CPU Counters" --output /tmp/gist.trace \
///     -- ./target/release/search_profile gist 100000
use std::time::Instant;

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

macro_rules! run_profile {
    ($data:ident, $queries:ident, $n:ident, $N:literal, $alpha:expr, $max_pts:ident) => {{
        use staged_diskann::{build_diskann_index, StagedDiskANN};

        eprintln!("Building index ({}-dim, {} points, alpha={})...", $N, $n, $alpha);
        let result = build_diskann_index(
            &$data, $n, $N, $alpha, 32, 48, false, None, None, true, 4,
        )
        .expect("build failed");
        let entry = result.entry_point;
        drop(result.index);

        let empty_ds = diskann::model::InmemDataset::<f32, $N>::new(0, 1.0).unwrap();
        let mut staged = StagedDiskANN::<$N>::new(
            empty_ds,
            &result.partitions,
            entry,
            32,
            4,
            None,
            None,
            None,
            false,
        );
        let mut ds = diskann::model::InmemDataset::<f32, $N>::new($n, 1.0).unwrap();
        ds.data.memcpy(&$data[..$n * $N]).unwrap();
        staged.dataset = ds;

        // Calibrate
        let calib_qs: Vec<[f32; $N]> = $queries[..200.min($queries.len())]
            .iter()
            .map(|q| {
                let mut a = [0f32; $N];
                a.copy_from_slice(&q[..$N]);
                a
            })
            .collect();
        let calib = staged.calibrate(&calib_qs, 48, 5).expect("calibrate");
        eprintln!(
            "Calibrated: threshold={:.2}, early_exit_limit={}",
            calib.threshold, calib.early_exit_limit
        );

        let queries_arr: Vec<[f32; $N]> = $queries
            .iter()
            .map(|q| {
                let mut a = [0f32; $N];
                a.copy_from_slice(&q[..$N]);
                a
            })
            .collect();

        // Warmup
        eprintln!("Warmup (100 queries)...");
        for q in queries_arr.iter().take(100) {
            let _ = staged.search(q, 10, 48, 5, calib.threshold, calib.early_exit_limit);
        }

        // Signal readiness via marker file, then sleep to let xctrace attach.
        let pid = std::process::id();
        let marker = format!("/tmp/search_profile_{}.ready", pid);
        std::fs::write(&marker, pid.to_string()).unwrap();
        eprintln!("PID={pid} — marker written to {marker}. Sleeping 5s for xctrace attach...");
        std::thread::sleep(std::time::Duration::from_secs(5));
        let _ = std::fs::remove_file(&marker);
        eprintln!("Starting profiled search section.");

        // ── Profiled section: only search, no build ──
        let search_ls = [32u32, 64, 128];
        let num_queries = queries_arr.len();
        let k = 10usize;
        let ws = 5usize;

        eprintln!(
            "\n=== SEARCH PROFILE: {}-dim, {} points, {} queries ===\n",
            $N, $n, num_queries
        );
        eprintln!(
            "{:<6} {:<12} {:<16} {:<16} {:<14}",
            "L", "QPS", "ns/query", "ns/dim_access", "total_ms"
        );

        for &l in &search_ls {
            let t = Instant::now();
            for q in &queries_arr {
                let _ = staged.search(q, k, l as usize, ws, calib.threshold, calib.early_exit_limit);
            }
            let elapsed = t.elapsed();
            let total_ms = elapsed.as_secs_f64() * 1000.0;
            let ns_per_query = elapsed.as_nanos() as f64 / num_queries as f64;
            // Rough estimate: each query visits ~L nodes, each expanding ~R neighbors,
            // each neighbor access = 1 distance computation = N dim reads.
            let est_dim_accesses = num_queries as f64 * l as f64 * 32.0;
            let ns_per_dim = elapsed.as_nanos() as f64 / est_dim_accesses;
            let qps = num_queries as f64 / elapsed.as_secs_f64();

            eprintln!(
                "L={:<4} {:<12.0} {:<16.0} {:<16.1} {:<14.1}",
                l, qps, ns_per_query, ns_per_dim, total_ms
            );
        }
        eprintln!();
    }};
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dataset = args.get(1).map(|s| s.as_str()).unwrap_or("sift");
    let max_points: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000);

    let (base_path, query_path, dim) = match dataset {
        "sift" => (
            "data/sift/sift_base.fvecs",
            "data/sift/sift_query.fvecs",
            128usize,
        ),
        "glove25" => (
            "data/glove25/glove-25-angular_base.fvecs",
            "data/glove25/glove-25-angular_query.fvecs",
            25usize,
        ),
        "glove100" => (
            "data/glove100/glove-100-angular_base.fvecs",
            "data/glove100/glove-100-angular_query.fvecs",
            100usize,
        ),
        "gist" => (
            "data/gist/gist_base.fvecs",
            "data/gist/gist_query.fvecs",
            960usize,
        ),
        _ => panic!("Unknown dataset: {dataset}. Use: sift, glove25, glove100, gist"),
    };

    eprintln!("Loading {dataset} (dim={dim}, max_points={max_points})...");
    let (data, n, _) = load_fvecs(base_path, max_points);
    let (qdata, nq, _) = load_fvecs(query_path, usize::MAX);
    let queries: Vec<Vec<f32>> = (0..nq)
        .map(|i| qdata[i * dim..(i + 1) * dim].to_vec())
        .collect();

    eprintln!("Loaded {n} base, {nq} queries.\n");

    // Theoretical cache analysis
    let vec_bytes = dim * 4;
    let cache_lines = (vec_bytes + 127) / 128;
    let dataset_mb = (n * vec_bytes) as f64 / 1_048_576.0;
    eprintln!("--- Cache Analysis ---");
    eprintln!("  Vector size:    {} bytes ({} cache lines)", vec_bytes, cache_lines);
    eprintln!("  Dataset size:   {:.1} MB", dataset_mb);
    eprintln!("  L1D capacity:   64 KB  ({:.0} vectors)", 65536.0 / vec_bytes as f64);
    eprintln!("  L2 capacity:    16 MB  ({:.0} vectors)", 16_777_216.0 / vec_bytes as f64);
    eprintln!(
        "  L2 fit ratio:   {:.1}%",
        (16_777_216.0 / (n * vec_bytes) as f64 * 100.0).min(100.0)
    );
    eprintln!();

    match dim {
        25 | 32 => run_profile!(data, queries, n, 32, 1.2f32, max_points),
        100 => run_profile!(data, queries, n, 100, 1.2f32, max_points),
        128 => run_profile!(data, queries, n, 128, 1.2f32, max_points),
        960 => run_profile!(data, queries, n, 960, 1.5f32, max_points),
        _ => panic!("Unsupported dimension: {dim}"),
    }
}
