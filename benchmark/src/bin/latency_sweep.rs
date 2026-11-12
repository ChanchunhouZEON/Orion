/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Explicit JSON configuration -> one resident Index -> request latency sweeps.
//! Relative input paths resolve beside the configuration, not the working directory.
use anyhow::{ensure, Context, Result};
use benchmark::latency::{
    config::Experiment,
    evaluation,
    report::{self, Point},
    runner,
};
use clap::Parser;
use orion_ann::{BaseInput, Index, OpenOptions, QueryBatch, SearchOptions};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde_json::json;
use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(about = "Measure single-query Index::search latency with closed-loop clients")]
struct Args {
    #[arg(long)]
    config: PathBuf,
    /// A new directory. Existing results are never overwritten.
    #[arg(long)]
    output: PathBuf,
}

fn resolve(base: &Path, path: &mut PathBuf) {
    if path.is_relative() {
        *path = base.join(&*path);
    }
}

fn run(args: &Args) -> Result<()> {
    let source = args.config.canonicalize()?;
    let mut experiment: Experiment = serde_json::from_reader(BufReader::new(File::open(&source)?))?;
    experiment.validate()?;
    let parent = source.parent().unwrap();
    for path in [
        &mut experiment.base,
        &mut experiment.query,
        &mut experiment.index.cache_dir,
    ] {
        resolve(parent, path);
    }
    for path in [
        &mut experiment.groundtruth,
        &mut experiment.snapshot,
        &mut experiment.index.graph.staged_file,
        &mut experiment.index.storage_options.directory,
    ]
    .into_iter()
    .flatten()
    {
        resolve(parent, path);
    }
    // Require prepared graph data: a typo must not launch a billion-point build.
    if let Some(snapshot) = &experiment.snapshot {
        ensure!(
            snapshot.is_dir() || snapshot.is_file(),
            "snapshot does not exist"
        );
    } else {
        ensure!(
            experiment
                .index
                .graph
                .staged_file
                .as_ref()
                .is_some_and(|p| p.is_file()),
            "latency sweep requires an existing staged graph or snapshot"
        );
    }
    std::fs::create_dir_all(args.output.parent().unwrap_or(Path::new(".")))?;
    std::fs::create_dir(&args.output).context("output directory must be new")?;
    let result_path = args.output.join("results.json");
    let mut report = json!({
        "schema": 1, "status": "loading", "config": experiment,
        "scope": "single-query Rust Index::search call; closed-loop load; not HTTP latency",
        "query_selection": "shared shuffled cycle; dynamic assignment to clients",
        "cache_policy": "steady-state, no cache flush", "histogram_significant_digits": 3,
        "histogram_max_ns": 3_600_000_000_000u64, "batch_size": 1,
        "timer": "std::time::Instant", "recall_protocol": "strict ID recall against ground-truth first k; no tie expansion",
        "feature_note": "benchmark feature includes existing search-stats counters",
        "qps_note": "instrumented closed-loop throughput, not historical batch-sweep QPS",
        "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
        // These describe the launch environment, not the binary's build.
        // A copied executable can outlive a checkout or toolchain switch.
        "checkout_git_head_at_run": command("git", &["rev-parse", "HEAD"]),
        "rustc_at_run": command("rustc", &["--version"]),
        "points": []
    });
    report::write_json(&result_path, &report)?;
    let outcome = (|| -> Result<()> {
        let began = Instant::now();
        let input = BaseInput::from_file(&experiment.base);
        let mut index = if let Some(snapshot) = &experiment.snapshot {
            OpenOptions {
                threads: Some(experiment.index.threads),
                auto_pin: Some(experiment.index.auto_pin),
                memory_budget_gib: experiment.index.memory_budget_gib,
                storage_options: Some(experiment.index.storage_options.clone()),
                ..Default::default()
            }
            .open(snapshot.as_path(), input)?
        } else {
            Index::new(input, experiment.index.clone())?
        };
        report["load_seconds"] = json!(began.elapsed().as_secs_f64());
        let queries = orion_ann::read_query_buffer(
            &experiment.query,
            experiment.query_format,
            index.dimension(),
        )?;
        let input = queries.as_input();
        let batch = QueryBatch::from_input(input, index.dimension())?;
        ensure!(!batch.is_empty(), "empty queries");
        let rows: Vec<_> = (0..batch.len())
            .map(|r| {
                QueryBatch::from_input(
                    input.slice(r * index.dimension()..(r + 1) * index.dimension()),
                    index.dimension(),
                )
            })
            .collect::<orion_ann::Result<_>>()?;
        experiment.protocol.min_requests = experiment.protocol.min_requests.max(rows.len());
        report["config"] = serde_json::to_value(&experiment)?;
        let calibration_start = Instant::now();
        // Recalibrate explicitly; a snapshot's historical settings must not
        // silently change the requested experimental protocol.
        let samples = experiment.calibration_samples.min(rows.len());
        index.calibrate_batch(
            QueryBatch::from_input(
                input.slice(0..samples * index.dimension()),
                index.dimension(),
            )?,
            experiment.k,
            Some(experiment.calibration_l),
        )?;
        report["calibration_seconds"] = json!(calibration_start.elapsed().as_secs_f64());
        report["index"] = index.info();
        report["unique_queries"] = json!(rows.len());
        report["query_element"] = json!(input.element_type());
        let truth = experiment
            .groundtruth
            .as_ref()
            .map(|path| -> Result<_> {
                let mut reader = orion_ann::benchmark::GroundTruthReader::new(
                    BufReader::new(File::open(path)?),
                    experiment.k,
                    index.len(),
                );
                let mut result = Vec::new();
                while let Some(row) = reader.next()? {
                    result.push(row);
                }
                evaluation::validate_truth(&result, rows.len(), experiment.k)?;
                Ok(result)
            })
            .transpose()?;
        report["status"] = json!("running");
        let mut points = Vec::new();
        for &beam in &experiment.beams {
            let options = SearchOptions::new(experiment.k, beam).with_scores(experiment.scores);
            let reference = index.search(batch, &options)?;
            let mut current = Vec::new();
            for &concurrency in &experiment.protocol.concurrency {
                let evaluation = evaluation::evaluate(
                    &index,
                    &rows,
                    &options,
                    &reference,
                    truth.as_deref(),
                    concurrency,
                )?;
                current.push(Point::new(beam, concurrency, evaluation));
            }
            for round in 0..experiment.protocol.rounds {
                let mut order: Vec<_> = (0..rows.len()).collect();
                order.shuffle(&mut StdRng::seed_from_u64(
                    experiment.protocol.seed.wrapping_add(round as u64),
                ));
                let mut conditions: Vec<_> = (0..current.len()).collect();
                if round % 2 == 1 {
                    conditions.reverse();
                }
                for position in conditions {
                    std::thread::sleep(Duration::from_secs_f64(
                        experiment.protocol.cooldown_seconds,
                    ));
                    let point = &mut current[position];
                    let trial = runner::measure(
                        &index,
                        &rows,
                        &options,
                        &order,
                        point.concurrency,
                        &experiment.protocol,
                        true,
                    )?;
                    let latency = trial.latency.as_ref().context("missing timed samples")?;
                    println!("{} L={} C={} round={} recall={:?} P50={:.3} P95={:.3} P99={:.3} us n={} complete={}", experiment.dataset, beam, point.concurrency, round + 1, point.evaluation.recall, latency.p50_us, latency.p95_us, latency.p99_us, trial.requests, trial.minimum_reached);
                    point.rounds.push(trial);
                    point.refresh()?;
                    report["points"] = serde_json::to_value(
                        points.iter().chain(current.iter()).collect::<Vec<_>>(),
                    )?;
                    report::write_json(&result_path, &report)?;
                }
            }
            if experiment.protocol.overhead_check {
                let mut order: Vec<_> = (0..rows.len()).collect();
                order.shuffle(&mut StdRng::seed_from_u64(experiment.protocol.seed));
                for point in &mut current {
                    point.untimed_control = Some(runner::measure(
                        &index,
                        &rows,
                        &options,
                        &order,
                        point.concurrency,
                        &experiment.protocol,
                        false,
                    )?);
                }
            }
            points.extend(current);
        }
        let complete = points
            .iter()
            .all(|p| p.rounds.iter().all(|r| r.minimum_reached));
        report["status"] = json!(if complete { "complete" } else { "incomplete" });
        report["points"] = serde_json::to_value(points)?;
        report::write_json(&result_path, &report)?;
        Ok(())
    })();
    if let Err(error) = &outcome {
        report["status"] = json!("failed");
        report["error"] = json!(format!("{error:#}"));
        report::write_json(&result_path, &report)?;
    }
    outcome
}

fn command(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_string())
}

fn main() -> Result<()> {
    env_logger::init();
    run(&Args::parse())
}
