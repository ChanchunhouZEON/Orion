//! EE-disabled same-graph neighbor-policy experiment.
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
use orion::algorithm::search::diagnostics::{NeighborMode, NoopObserver, SearchTrace};
use rayon::prelude::*;
use serde_json::{json, Value};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
#[command(name = "adaptive")]
struct AdaptiveArgs {
    #[command(flatten)]
    run: cli::config::Args,
    /// Evenly spaced diagnostic queries; 0 means all. Timing uses all queries.
    #[arg(long, default_value_t = 1000)]
    diagnostic_queries: usize,
    #[arg(long)]
    output: Option<PathBuf>,
    /// Best measured QPS meeting each target; no interpolation or extrapolation.
    #[arg(long, value_delimiter = ',', default_value = "0.9,0.95,0.99")]
    recall_targets: Vec<f64>,
}

fn recall(ids: &[u32], truth: &[u32]) -> f64 {
    ids.iter().filter(|id| truth.contains(id)).count() as f64 / truth.len() as f64
}

fn paired_discovery(a: &[Vec<Option<usize>>], b: &[Vec<Option<usize>>]) -> Value {
    let (mut count, mut left, mut right) = (0usize, 0usize, 0usize);
    for (a, b) in a.iter().zip(b) {
        for (x, y) in a.iter().zip(b) {
            if let (Some(x), Some(y)) = (x, y) {
                count += 1;
                left += x;
                right += y;
            }
        }
    }
    json!({
        "common_targets":   count,
        "left_mean_step":   (count > 0).then(|| left as f64 / count as f64),
        "right_mean_step":  (count > 0).then(|| right as f64 / count as f64),
    })
}

fn run<const N: usize>(
    args: &AdaptiveArgs,
    config: &cli::config::ResolvedRunConfig,
    mut data: cli::data::LoadedDataset,
) -> Result<(), String>
where
    [f32; N]: vector::FullPrecisionDistance<f32, N>,
{
    let idx = cli::index::load_index::<N>(config, &mut data)?;
    if config.prepare_only { return Ok(()); }
    let queries: Vec<[f32; N]> = data
        .queries
        .iter()
        .map(|q| q.as_slice().try_into().expect("validated dimension"))
        .collect();
    let settings = &config.sweep;
    if queries.is_empty() || settings.threads > orion::algorithm::search::MAX_WORKERS {
        return Err("Queries must be nonempty and threads must not exceed MAX_WORKERS".into());
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(settings.threads)
        .build()
        .map_err(|e| e.to_string())?;

    let calib = pool
        .install(|| {
            idx.calibrate(
                &queries[..settings.calibration_samples.min(queries.len())],
                settings.calibration_l,
                config.window_size,
                orion::CalibrationConfig {
                    k: settings.k,
                    metric: config.metric.calibration_metric(),
                },
            )
        })
        .map_err(|e| e.to_string())?;

    let pf = cascade::build_prefilter(&idx, config.cascade.prefilter, config.cascade.admission);
    let ad = cascade::build_admission(&idx, config.cascade.admission);
    let rr = cascade::build_rerank(&idx, config.cascade.rerank);
    let count = if args.diagnostic_queries == 0 {
        queries.len()
    } else {
        args.diagnostic_queries.min(queries.len())
    };

    let sample: Vec<usize> = (0..count).map(|i| i * queries.len() / count).collect();
    let mut points = Vec::new();

    // Prohibit early exit by setting consecutive no admit tolerant to usize::MAX
    for &l in &settings.search_list_sizes {
        let search = |qi: usize, mode, observer: &mut NoopObserver| {
            idx.search_unified_observed(
                &queries[qi],
                settings.k,
                l,
                config.window_size,
                calib.threshold,
                // consecutive no admit tolerant
                usize::MAX,
                pf.as_deref(),
                ad.as_ref(),
                rr.as_ref(),
                mode,
                observer,
            )
            .map_err(|e| e.to_string())
        };
        let mut arms = Vec::new();
        let mut discoveries = Vec::new();
        // Instrumentation pass is separate from all timed trials.
        for mode in NeighborMode::ALL {
            let rows: Result<Vec<_>, String> = pool.install(|| {
                sample
                    .par_iter()
                    .map(|&qi| {
                        let truth = &data.ground_truth[qi][..settings.k];

                        let mut trace = SearchTrace::new(truth);

                        let ids = idx
                            .search_unified_observed(
                                &queries[qi],
                                settings.k,
                                l,
                                config.window_size,
                                calib.threshold,
                                usize::MAX,
                                pf.as_deref(),
                                ad.as_ref(),
                                rr.as_ref(),
                                mode,
                                &mut trace,
                            )
                            .map_err(|e| e.to_string())?;

                        if ids != search(qi, mode, &mut NoopObserver)? {
                            return Err(format!(
                                "Trace changed results: query={qi}, L={l}, mode={mode:?}"
                            ));
                        }

                        let rerank_ndc = if config.cascade.rerank == cascade::RerankChoice::None {
                            0
                        } else {
                            trace.rerank_candidates
                        };

                        let discovered =
                            trace.first_discovery.iter().filter(|v| v.is_some()).count();

                        let row = json!({
                            "query":                    qi,
                            "recall":                   recall(&ids, truth),
                            "expansions":               trace.expansions,
                            "pre_expansions":           trace.pre_expansions,
                            "post_expansions":          trace.post_expansions,
                            "first_switch_step":        trace.first_switch_step,
                            "reversals":                trace.reversals,
                            "unique_candidates":        trace.unique_candidates,
                            "admission_ndc":            trace.admission_ndc,
                            "prefilter_candidate_ndc":  trace.prefilter_candidate_ndc,
                            "prefilter_threshold_ndc":  trace.prefilter_threshold_ndc,
                            "rerank_ndc":               rerank_ndc,
                            "total_stage_ndc":
                                trace.admission_ndc +
                                trace.prefilter_candidate_ndc +
                                trace.prefilter_threshold_ndc +
                                rerank_ndc,
                            "target_ids": truth,
                            "first_discovery":          trace.first_discovery,
                            "first_admission":          trace.first_admission,
                            "discovery_fraction":       discovered as f64 / settings.k as f64,
                        });
                        Ok((row, trace.first_discovery))
                    })
                    .collect()
            });
            let rows = rows?;
            let discovery = rows
                .iter()
                .map(|(_, times)| times.clone())
                .collect::<Vec<_>>();
            let rows = rows.into_iter().map(|(row, _)| row).collect::<Vec<_>>();
            let mut means = serde_json::Map::new();
            for key in [
                "recall",
                "expansions",
                "pre_expansions",
                "post_expansions",
                "unique_candidates",
                "admission_ndc",
                "prefilter_candidate_ndc",
                "prefilter_threshold_ndc",
                "rerank_ndc",
                "total_stage_ndc",
                "discovery_fraction",
                "reversals",
            ] {
                means.insert(
                    key.into(),
                    json!(
                        rows.iter().map(|r| r[key].as_f64().unwrap()).sum::<f64>()
                            / rows.len() as f64
                    ),
                );
            }
            discoveries.push(discovery);
            arms.push(json!(
            {
                "mode": mode.name(),
                "diagnostic_means": means,
                "queries": rows
            }));
        }

        // Iterate three different neighbor modes with collecting the `qps` and `recalls`.
        let batch = |mode| -> Result<Vec<Vec<u32>>, String> {
            pool.install(|| {
                (0..queries.len())
                    .into_par_iter()
                    .map(|qi| search(qi, mode, &mut NoopObserver))
                    .collect()
            })
        };
        for mode in NeighborMode::ALL {
            batch(mode)?;
        }

        let mut qps = [Vec::new(), Vec::new(), Vec::new()];
        let mut recalls = [0.0; 3];
        // Rotate arm order across trials; use the same cache-flush policy in all arms.
        for trial in 0..settings.trials {
            for offset in 0..3 {
                let arm = (trial + offset) % 3;
                utils::flush_cache();
                let start = Instant::now();
                let ids = batch(NeighborMode::ALL[arm])?;
                let elapsed = start.elapsed().as_secs_f64();
                qps[arm].push(queries.len() as f64 / elapsed);
                recalls[arm] = ids
                    .iter()
                    .zip(&data.ground_truth)
                    .map(|(ids, gt)| recall(ids, &gt[..settings.k]))
                    .sum::<f64>()
                    / queries.len() as f64;
            }
        }
        for arm in 0..3 {
            let mut sorted = qps[arm].clone();
            sorted.sort_by(f64::total_cmp);
            arms[arm]["qps_trials"] = json!(qps[arm]);
            arms[arm]["qps_median"] = json!(sorted[sorted.len() / 2]);
            arms[arm]["recall"] = json!(recalls[arm]);
            log::info!(
                "L={l} {} R@{}={:.6} QPS={:.0}",
                NeighborMode::ALL[arm].name(),
                settings.k,
                recalls[arm],
                sorted[sorted.len() / 2]
            );
        }

        points.push(json!(
            {
                "l": l,
                "arms": arms,
                "paired_discovery": {
                    "full_vs_local":    paired_discovery(&discoveries[0], &discoveries[1]),
                    "local_vs_extra":   paired_discovery(&discoveries[1], &discoveries[2]),
                    "full_vs_extra":    paired_discovery(&discoveries[0], &discoveries[2]),
                }
            }
        ));
    }

    let out = args.output.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "visualizations/adaptive_{}_k{}.json",
            config.dataset.as_deref().unwrap_or("custom"),
            settings.k
        ))
    });
    let matched_recall: Vec<_> = args
        .recall_targets
        .iter()
        .map(|&target| {
            let arms: Vec<_> = (0..3)
                .map(|arm| {
                    let best = points.iter()
                        .filter(|point| point["arms"][arm]["recall"].as_f64().unwrap() >= target)
                        .max_by(|a, b|
                            a["arms"][arm]["qps_median"]
                                .as_f64()
                                .unwrap()
                            .total_cmp(
                                &b["arms"][arm]["qps_median"]
                                    .as_f64()
                                    .unwrap()));
                json!({
                    "mode": NeighborMode::ALL[arm].name(),
                    "best_measured": best.map(
                        |p| json!(
                            {
                                "l": p["l"],
                                "recall": p["arms"][arm]["recall"],
                                "qps": p["arms"][arm]["qps_median"],
                                "diagnostic_means": p["arms"][arm]["diagnostic_means"],
                            }
                        )
                    )})
                })
                .collect();
            json!({"target": target, "arms": arms})
        })
        .collect();
    let result = json!({
        "schema_version": 1,
        "config": config,
        "num_points": data.num_points,
        "query_count": queries.len(),
        "diagnostic_query_ids": sample,
        "threshold": calib.threshold,
        "early_exit": false,
        "rerank_factor": 2,
        "semantics": {
            "comparison": "Only post-convergence neighbors differ; all arms retain convergence-dependent flush scheduling",
            "discovery": "First visited-set-deduplicated encounter before prefilter; entry is step 0; null means never found",
            "admission": "First admission-cutoff success, not guaranteed retained PQ membership",
            "ndc": "Stage distance evaluations, including entry, prefilter threshold recomputation, and final rerank; not unique objects",
            "paired": "Means use only targets discovered by both arms on the same sampled query",
            "timing": "No-op observer, all queries, separate trials; existing production counters remain enabled"
        },
        "points": points,
        "matched_recall": matched_recall,
    });
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&result).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    log::info!("Saved {}", out.display());
    Ok(())
}

fn execute(args: AdaptiveArgs) -> Result<(), String> {
    if args.recall_targets.is_empty()
        || args
            .recall_targets
            .iter()
            .any(|r| !r.is_finite() || *r <= 0.0 || *r > 1.0)
    {
        return Err("recall-targets must be finite values in (0, 1]".into());
    }
    let config = args.run.resolve()?;
    if args.run.print_config {
        println!(
            "{}",
            serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if args.run.preflight {
        println!("{}", serde_json::to_string_pretty(&cli::resources::preflight_report(&config)?).map_err(|e| e.to_string())?);
        return Ok(());
    }
    let data = cli::data::LoadedDataset::load(&config)?;
    with_supported_dimension!(config.dimension, |D| run::<D>(&args, &config, data))
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    if let Err(e) = execute(AdaptiveArgs::parse()) {
        log::error!("{e}");
        std::process::exit(2);
    }
}
