//! Per-query prefilter trajectories. Instrumented diagnostics, never QPS timings.
use clap::Parser;
use orion::algorithm::search::diagnostics::{NeighborMode, NoopObserver, SearchObserver};
use orion_cli::{cascade, cli, with_supported_dimension};
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    run: cli::config::Args,
    /// Evenly spaced queries; zero selects all queries.
    #[arg(long, default_value_t = 0)]
    diagnostic_queries: usize,
    #[arg(long, default_value_t = 20)]
    progress_bins: usize,
    #[arg(long)]
    disable_ee: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Default)]
struct Trace {
    rows: Vec<Value>,
    first_switch: Option<usize>,
    early_exit: bool,
}
impl SearchObserver for Trace {
    fn expansion(&mut self, converged: bool, ids: &[u32]) {
        let step = self.rows.len() + 1;
        if converged {
            self.first_switch.get_or_insert(step);
        }
        self.rows
            .push(json!({"step": step, "converged": converged, "unseen": ids.len()}));
    }
    fn prefilter_outcome(
        &mut self,
        cutoff: Option<f32>,
        admission: Option<f32>,
        tested: usize,
        rejected: usize,
    ) {
        let row = self.rows.last_mut().expect("expansion precedes filtering");
        row["prefilter_cutoff"] = json!(cutoff);
        row["admission_cutoff"] = json!(admission);
        row["tested"] = json!(tested);
        row["rejected"] = json!(rejected);
        row["reject_rate"] = json!((tested > 0).then(|| rejected as f64 / tested as f64));
    }
    fn early_exit(&mut self) {
        self.early_exit = true;
    }
}

fn summarize(rows: &[Value]) -> Value {
    let tested: u64 = rows.iter().map(|r| r["tested"].as_u64().unwrap()).sum();
    let rejected: u64 = rows.iter().map(|r| r["rejected"].as_u64().unwrap()).sum();
    let cutoffs: Vec<f64> = rows
        .iter()
        .filter_map(|r| r["prefilter_cutoff"].as_f64())
        .collect();
    json!({"expansions": rows.len(), "tested": tested, "rejected": rejected,
        "reject_rate": (tested > 0).then(|| rejected as f64 / tested as f64),
        "mean_prefilter_cutoff": (!cutoffs.is_empty()).then(|| cutoffs.iter().sum::<f64>() / cutoffs.len() as f64)})
}

fn run<const D: usize>(
    args: &Args,
    config: &cli::config::ResolvedRunConfig,
    mut data: cli::data::LoadedDataset,
) -> Result<(), String>
where
    [f32; D]: vector::FullPrecisionDistance<f32, D>,
{
    let idx = cli::index::load_index::<D>(config, &mut data)?;
    if config.prepare_only {
        return Ok(());
    }
    let queries: Vec<[f32; D]> = data
        .queries
        .iter()
        .map(|q| q.as_slice().try_into().expect("validated dimension"))
        .collect();
    if queries.is_empty() {
        return Err("No queries".into());
    }
    let s = &config.sweep;
    let calib = idx
        .calibrate(
            &queries[..s.calibration_samples.min(queries.len())],
            s.calibration_l,
            config.window_size,
            orion::CalibrationConfig {
                k: s.k,
                metric: config.metric.calibration_metric(),
            },
        )
        .map_err(|e| e.to_string())?;
    let c = config.search_plan().cascade();
    let pf = cascade::build_prefilter(&idx, c.prefilter, c.admission)
        .ok_or("This experiment requires an enabled prefilter; use --prefilter jl")?;
    let ad = cascade::build_admission(&idx, c.admission).map_err(|e| e.to_string())?;
    let rr = cascade::build_rerank(&idx, c.rerank);
    let limit = if args.disable_ee {
        usize::MAX
    } else {
        calib.early_exit_limit
    };
    let count = if args.diagnostic_queries == 0 {
        queries.len()
    } else {
        args.diagnostic_queries.min(queries.len())
    };
    let mut points = Vec::new();
    for &l in &s.search_list_sizes {
        let mut records = Vec::new();
        for qi in (0..count).map(|i| i * queries.len() / count) {
            let mut trace = Trace::default();
            let ids = idx
                .search_unified_observed(
                    &queries[qi],
                    s.k,
                    l,
                    config.window_size,
                    calib.threshold,
                    limit,
                    Some(pf.as_ref()),
                    ad.as_ref(),
                    rr.as_ref(),
                    NeighborMode::LocalExtra,
                    &mut trace,
                )
                .map_err(|e| e.to_string())?;
            let control = idx
                .search_unified_observed(
                    &queries[qi],
                    s.k,
                    l,
                    config.window_size,
                    calib.threshold,
                    limit,
                    Some(pf.as_ref()),
                    ad.as_ref(),
                    rr.as_ref(),
                    NeighborMode::LocalExtra,
                    &mut NoopObserver,
                )
                .map_err(|e| e.to_string())?;
            if ids != control {
                return Err(format!("Observer changed results: query={qi}, L={l}"));
            }
            let n = trace.rows.len();
            let bins: Vec<_> = (0..args.progress_bins)
                .map(|b| {
                    let start = b * n / args.progress_bins;
                    let end = (b + 1) * n / args.progress_bins;
                    summarize(&trace.rows[start..end])
                })
                .collect();
            let recall = data
                .ground_truth
                .as_ref()
                .and_then(|gt| gt.get(qi))
                .filter(|gt| gt.len() >= s.k)
                .map(|gt| {
                    ids.iter().filter(|id| gt[..s.k].contains(id)).count() as f64 / s.k as f64
                });
            records.push(
                json!({"query": qi, "recall": recall, "first_switch_step": trace.first_switch,
                "early_exited": trace.early_exit, "overall": summarize(&trace.rows),
                "last_quarter": summarize(&trace.rows[n * 3 / 4..]),
                "bins": bins, "steps": trace.rows}),
            );
        }
        log::info!("Completed L={l}, {count} queries (instrumented, result equivalence checked)");
        points.push(json!({"l": l, "queries": records}));
    }
    let out = args.output.clone().unwrap_or_else(|| {
        PathBuf::from(format!(
            "visualizations/prefilter_progress_{}_k{}.json",
            config.dataset.as_str(),
            s.k
        ))
    });
    if let Some(parent) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let result = json!({"schema_version": 1, "config": config, "query_count": queries.len(),
        "diagnostic_queries": count, "progress_bins": args.progress_bins,
        "calibrated_tau": calib.threshold, "calibrated_lambda": calib.early_exit_limit,
        "early_exit_enabled": !args.disable_ee,
        "semantics": {"progress": "Per-query expansion count, including bypassed stages; bins use floor boundaries",
            "reject_rate": "Rejected / actually tested, after visited-set deduplication; null for zero tests",
            "cutoff": "Raw prefilter score cutoff; cumulative mean of cached beam sketch means times slack, not EMA",
            "bypass": "Prefilter cutoff null during peeled expansions or before beam full; admission cutoff null until beam full",
            "timing": "Sequential diagnostic queries, no throughput measurements; every result checked against no-op observer"},
        "points": points});
    std::fs::write(
        &out,
        serde_json::to_vec(&result).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    log::info!("Saved {}", out.display());
    Ok(())
}

fn execute(args: Args) -> Result<(), String> {
    if args.progress_bins == 0 {
        return Err("progress-bins must be positive".into());
    }
    let config = args.run.resolve_for_sweep()?;
    if args.run.print_config {
        println!(
            "{}",
            serde_json::to_string_pretty(&config).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if args.run.preflight {
        println!(
            "{}",
            serde_json::to_string_pretty(&cli::resources::preflight_report(&config)?)
                .map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let data = cli::data::LoadedDataset::load(&config)?;
    with_supported_dimension!(config.dimension, |D| run::<D>(&args, &config, data))
}

fn main() {
    let _ = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .try_init();
    if let Err(e) = execute(Args::parse()) {
        log::error!("{e}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_windows_are_missing_not_zero() {
        assert!(summarize(&[])["reject_rate"].is_null());
        let mut t = Trace::default();
        t.expansion(false, &[1, 2]);
        t.prefilter_outcome(None, None, 0, 0);
        assert!(summarize(&t.rows)["reject_rate"].is_null());
    }
    #[test]
    fn counts_are_weighted_within_each_query_window() {
        let mut t = Trace::default();
        t.expansion(false, &[1, 2, 3, 4]);
        t.prefilter_outcome(Some(20.0), Some(1.0), 4, 1);
        t.expansion(true, &[5]);
        t.prefilter_outcome(Some(18.0), Some(0.8), 1, 1);
        assert_eq!(summarize(&t.rows)["reject_rate"], json!(0.4));
        assert_eq!(t.first_switch, Some(2));
    }
}
