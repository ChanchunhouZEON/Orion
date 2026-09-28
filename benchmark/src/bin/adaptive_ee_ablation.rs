//! Same-graph 2x2 neighbor-switching / early-exit experiment.
#[path = "../runner/cascade.rs"]
mod cascade;
#[path = "../cli/mod.rs"]
mod cli;
#[path = "../config.rs"]
mod config;
#[path = "../runner/parlayann_bridge.rs"]
mod parlayann_bridge;
#[path = "../utils.rs"]
mod utils;

use clap::Parser;
use orion::algorithm::search::diagnostics::{NeighborMode, NoopObserver, PhaseWork, SynergyTrace};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(name = "adaptive_ee_ablation")]
struct AdaptiveEeAblationArgs {
    #[command(flatten)]
    run: cli::config::Args,
    /// Evenly spaced diagnostic queries; 0 (default) means all queries.
    #[arg(long, default_value_t = 0)]
    diagnostic_queries: usize,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Minimum recall for discrete NDC/QPS selections; no interpolation.
    #[arg(long, value_delimiter = ',', default_value = "0.9,0.95,0.99")]
    recall_targets: Vec<f64>,
    /// Override the shared calibrated EE limit for a controlled experiment.
    #[arg(long)]
    ee_limit: Option<usize>,
}

#[derive(Clone, Copy)]
struct Arm {
    name: &'static str,
    mode: NeighborMode,
    ee: bool,
}

const ARMS: [Arm; 4] = [
    Arm { name: "full",         mode: NeighborMode::FullNeighbor,   ee: false },
    Arm { name: "adaptive",     mode: NeighborMode::LocalExtra,     ee: false },
    Arm { name: "ee_only",      mode: NeighborMode::FullNeighbor,   ee: true },
    Arm { name: "adaptive_ee",  mode: NeighborMode::LocalExtra,     ee: true },
];

fn recall(ids: &[u32], truth: &[u32]) -> f64 {
    ids.iter().filter(|id| truth.contains(id)).count() as f64 / truth.len() as f64
}

fn phase_json(phase: &PhaseWork) -> Value {
    json!({
        "expansions": phase.expansions,
        "admission_ndc": phase.admission_ndc,
        "prefilter_candidate_ndc": phase.prefilter_candidate_ndc,
        "prefilter_threshold_ndc": phase.prefilter_threshold_ndc,
        "ndc": phase.ndc(),
    })
}

fn selection(point: &Value, arm: usize) -> Value {
    let a = &point["arms"][arm];
    json!({
        "l": point["l"],
        "recall": a["recall"],
        "qps": a["qps_median"],
        "diagnostic_means": a["diagnostic_means"]})
}

fn run<const N: usize>(
    args: &AdaptiveEeAblationArgs,
    config: &cli::config::ResolvedRunConfig,
    mut data: cli::data::LoadedDataset,
) -> Result<(), String>
where
    [f32; N]: vector::FullPrecisionDistance<f32, N>,
{
    let settings = &config.sweep;
    let output = args
        .output
        .clone()
        .unwrap_or_else(
            || PathBuf::from(format!("visualizations/adaptive_ee_ablation_{}_k{}.json",
               config.dataset.as_deref().unwrap_or("custom"), settings.k, )));

    if output.exists() {
        return Err(format!("Refusing to overwrite {}", output.display()));
    }

    if settings.threads == 0 || settings.threads > orion::algorithm::search::MAX_WORKERS
        || settings.trials == 0 || settings.search_list_sizes.is_empty()
        || settings.calibration_samples == 0
    {
        return Err("Invalid thread, trial, calibration sample count or empty L sweep".into());
    }

    let queries: Vec<[f32; N]> = data
        .queries
        .iter()
        .map(|q| q.as_slice().try_into().expect("validated dimension"))
        .collect();
    if queries.is_empty() || settings.k == 0 || data.ground_truth.len() != queries.len()
        || data.ground_truth.iter().any(|gt| gt.len() < settings.k)
    {
        return Err("Nonempty queries and at least k ground-truth targets per query are required".into());
    }


    let idx = cli::index::load_index::<N>(config, &mut data)?;
    if config.prepare_only { return Ok(()); }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(settings.threads)
        .build()
        .map_err(|e| e.to_string())?;
    let calib = pool.install(|| idx.calibrate(
        &queries[..settings.calibration_samples.min(queries.len())],
        settings.calibration_l, config.window_size,
        orion::CalibrationConfig { k: settings.k, metric: config.metric.calibration_metric() },
    )).map_err(|e| e.to_string())?;

    let ee_limit = args.ee_limit.unwrap_or(calib.early_exit_limit);
    log::info!("Shared calibration: threshold={}, EE limit={ee_limit}", calib.threshold);

    let pf = cascade::build_prefilter(&idx, config.cascade.prefilter, config.cascade.admission);
    let ad = cascade::build_admission(&idx, config.cascade.admission);
    let rr = cascade::build_rerank(&idx, config.cascade.rerank);

    let count =
        if args.diagnostic_queries == 0 { queries.len() }
        else { args.diagnostic_queries.min(queries.len()) };
    let sample: Vec<usize> = (0..count).map(|i| i * queries.len() / count).collect();
    let mut points = Vec::new();

    for &l in &settings.search_list_sizes {
        let search = |qi: usize, arm: Arm| {
            idx.search_unified_observed(&queries[qi], settings.k, l, config.window_size,
                calib.threshold, if arm.ee { ee_limit } else { usize::MAX },
                pf.as_deref(), ad.as_ref(), rr.as_ref(), arm.mode, &mut NoopObserver)
                .map_err(|e| e.to_string())
        };
        let mut arms = Vec::new();
        for arm in ARMS {
            let rows: Result<Vec<Value>, String> = pool.install(|| sample.par_iter().map(|&qi| {
                let truth = &data.ground_truth[qi][..settings.k];
                let mut trace = SynergyTrace::default();
                let ids = idx
                    .search_unified_observed(
                        &queries[qi],
                        settings.k,
                        l,
                        config.window_size,
                        calib.threshold,
                        if arm.ee { ee_limit } else { usize::MAX },
                        pf.as_deref(), ad.as_ref(), rr.as_ref(),
                        arm.mode,
                        &mut trace)
                    .map_err(|e| e.to_string())?;

                if ids != search(qi, arm)? {
                    return Err(format!("Observer changed results: query={qi}, L={l}, arm={}", arm.name));
                }

                let rerank = if config.cascade.rerank == cascade::RerankChoice::None { 0 }
                    else { trace.rerank_candidates };

                Ok(json!({
                    "query": qi,
                    "recall": recall(&ids, truth),
                    "first_converged_step": trace.first_converged_step,
                    "early_exited": trace.early_exited,
                    "before": phase_json(&trace.before),
                    "after": phase_json(&trace.after),
                    "before_ndc": trace.before.ndc(),
                    "after_ndc": trace.after.ndc(),
                    "before_expansions": trace.before.expansions,
                    "after_expansions": trace.after.expansions,
                    "rerank_ndc": rerank,
                    "total_stage_ndc": trace.before.ndc() + trace.after.ndc() + rerank,
                }))
            }).collect());

            let rows = rows?;
            let mut means = serde_json::Map::new();

            // Metrics presented in the final json record just multiplex the `key` name.
            for key in [
                "recall",
                "before_ndc",
                "after_ndc",
                "before_expansions",
                "after_expansions",
                "rerank_ndc",
                "total_stage_ndc"] {
                means.insert(
                    key.into(),
                    json!(
                        rows
                        .iter()
                        .map(|r| r[key].as_f64().unwrap())
                    .sum::<f64>() / count as f64));
            }

            // Metrics need converting into target `key`.
            for (key, field) in [
                ("early_exit_fraction", "early_exited"),
                ("converged_fraction", "first_converged_step")] {
                let number = rows.
                    iter()
                    .filter(|r|
                        if field == "early_exited" {
                            r[field].as_bool().unwrap()
                        }
                        else { !r[field].is_null() } )
                    .count();
                means.insert(key.into(), json!(number as f64 / count as f64));
            }

            arms.push(json!(
                {
                    "arm": arm.name,
                    "neighbor_mode": arm.mode.name(),
                    "early_exit": arm.ee,
                    "diagnostic_means": means,
                    "queries": rows}));
        }

        // Validation check: all arms must share the prefix through the first convergence decision.
        for i in 0..count {
            for arm in 1..4 {
                let base = &arms[0]["queries"][i];
                let row = &arms[arm]["queries"][i];
                if base["before"] != row["before"]
                    || base["first_converged_step"] != row["first_converged_step"] {
                    return Err(format!("Prefix mismatch: query={}, L={l}", sample[i]));
                }
            }
        }
        let batch = |arm| -> Result<Vec<Vec<u32>>, String> {
            pool.install(||
                (0..queries.len())
                    .into_par_iter()
                    .map(|qi| search(qi, arm))
                    .collect())
        };

        for arm in ARMS { batch(arm)?; }
        let mut qps: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
        let mut recalls = [0.0; 4];
        for trial in 0..settings.trials {
            for offset in 0..4 {
                let a = (trial + offset) % 4;
                utils::flush_cache();
                let start = Instant::now();
                let ids = batch(ARMS[a])?;
                qps[a].push(queries.len() as f64 / start.elapsed().as_secs_f64());
                recalls[a] = ids.iter().zip(&data.ground_truth)
                    .map(|(ids, gt)|
                        recall(ids, &gt[..settings.k]))
                    .sum::<f64>() / queries.len() as f64;
            }
        }
        for a in 0..4 {
            let mut sorted = qps[a].clone();
            sorted.sort_by(f64::total_cmp);
            let median = (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0;
            arms[a]["qps_trials"] = json!(qps[a]);
            arms[a]["qps_median"] = json!(median);
            arms[a]["recall"] = json!(recalls[a]);
            log::info!("L={l} {} recall={:.6} QPS={median:.0} NDC={}",
                ARMS[a].name, recalls[a], arms[a]["diagnostic_means"]["total_stage_ndc"]);
        }

        points.push(json!({"l": l, "arms": arms}));
    }

    let matched: Vec<_> = args.recall_targets.iter().map(|&target| {
        let arms: Vec<_> = (0..4).map(|arm| {
            let eligible: Vec<_> =
                points
                    .iter()
                    .filter(|p|
                        p["arms"][arm]["recall"].as_f64().unwrap() >= target)
                    .collect();

            let min_ndc =
                eligible
                    .iter()
                    .min_by(|a, b|
                            a["arms"][arm]["diagnostic_means"]["total_stage_ndc"]
                                .as_f64()
                                .unwrap()
                        .total_cmp(
                            &b["arms"][arm]["diagnostic_means"]["total_stage_ndc"]
                                .as_f64()
                                .unwrap()
                        ));

            let max_qps =
                eligible
                    .iter()
                    .max_by(|a, b|
                        a["arms"][arm]["qps_median"].as_f64().unwrap()
                .total_cmp(
                        &b["arms"][arm]["qps_median"].as_f64().unwrap()
                ));

            json!({
                "arm": ARMS[arm].name,
                "min_ndc_measured": min_ndc.map(|p| selection(p, arm)),
                "max_qps_measured": max_qps.map(|p| selection(p, arm))})
        }).collect();
        json!({"target": target, "arms": arms})
    }).collect();
    let result = json!({
        "schema_version": 1,
        "experiment": "adaptive_ee_ablation",
        "config": config,
        "num_points": data.num_points,
        "query_count": queries.len(),
        "diagnostic_query_ids": sample,
        "threshold": calib.threshold,
        "calibrated_early_exit_limit": calib.early_exit_limit,
        "early_exit_limit": ee_limit,
        "rerank_factor": 2,
        "semantics": {
            "comparison": "2x2 neighbor switching and EE; all arms retain admission convergence and convergence-dependent flush scheduling",
            "phase": "First converged expansion starts the tail, which includes any later reversals; entry is prefix; final rerank is separate",
            "ndc": "Sum of stage distance evaluations after visited-set candidate deduplication, including repeated prefilter threshold work; not unique objects or equal-cost operations",
            "early_exited": "The EE condition triggered a loop break; this does not prove the frontier would otherwise have continued",
            "timing": "All queries, no-op observer, separate warmed trials with rotated arm order and identical cache flushing",
            "matched_recall": "Discrete measured configurations meeting a minimum recall, not exactly equal recall; NDC is averaged over diagnostic queries; unreachable targets are null",
        },
        "points": points, "matched_recall": matched,
    });

    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)
        .map_err(|e| e.to_string())?;

    file.write_all(&serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    log::info!("Saved {}", output.display());

    Ok(())
}

fn execute(args: AdaptiveEeAblationArgs) -> Result<(), String> {
    if args.recall_targets.is_empty() || args.recall_targets.iter()
        .any(|r| !r.is_finite() || *r <= 0.0 || *r > 1.0) {
        return Err("recall-targets must be finite values in (0, 1]".into());
    }
    if matches!(args.ee_limit, Some(0 | usize::MAX)) {
        return Err("ee-limit must be in [1, usize::MAX)".into());
    }
    let config = args.run.resolve()?;
    if args.run.print_config {
        println!("{}", serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?);
        return Ok(());
    }
    if args.run.preflight {
        println!("{}", serde_json::to_string_pretty(&cli::resources::report(&config)?).map_err(|e| e.to_string())?);
        return Ok(());
    }
    let data = cli::data::LoadedDataset::load(&config)?;
    with_supported_dimension!(config.dimension, |D| run::<D>(&args, &config, data))
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    if let Err(error) = execute(AdaptiveEeAblationArgs::parse()) {
        log::error!("{error}");
        std::process::exit(2);
    }
}
