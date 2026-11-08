/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Ordinary one-pass search.
//!
//! This binary performs a single search run over a query stream.
//! Replay, thermal controls, parameter sweeps, and other benchmarking
//! machinery live in the benchmark binaries.

use clap::Parser;
mod args;
mod resolve;
use args::Args;

use orion_ann::execution::{
    open_queries, open_snapshot_queries, resources, F32CascadeBackend, GroundTruthReader,
    LoadedBase, NativeU8L2Backend, QuerySource, ResolvedRunConfig, SearchBackend,
};
use orion_ann::{Error, Result};
use std::{
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(
    name = "orion",
    about = "Build or search an Orion index; ground truth is optional"
)]
struct Command {
    #[command(flatten)]
    run: Args,

    /// Open a published Index::save snapshot and use its saved calibration.
    #[arg(
        long, value_name = "DIR", requires_all = ["base", "query"],
        conflicts_with_all = [
            "cache_dir", "staged_file",
            "graph_source", "graph_degree", "build_l", "alpha", "max_extra",
            "max_points", "window_size", "base_format", "vector_storage", "dimension",
            "metric", "prefilter", "admission", "rerank", "print_config", "preflight",
            "prepare_only"
        ]
    )]
    snapshot: Option<PathBuf>,

    /// Single beam width; defaults to max(k, 64).
    /// Sweeps use `orion-sweep`.
    #[arg(long, alias = "ls", alias = "search-list-sizes")]
    search_l: Option<usize>,

    /// Write result IDs as JSONL to a new file.
    /// Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Structured errors on stderr for services; success output is unchanged.
    #[arg(long, value_enum, default_value = "text")]
    error_format: ErrorFormat,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum ErrorFormat {
    Text,
    Json,
}

fn run<B: SearchBackend>(config: &ResolvedRunConfig, output: Option<&PathBuf>) -> Result<()> {
    // Open lightweight query/evaluation inputs before allocating the potentially
    // much larger base dataset and index structures.
    let gt_file = if config.prepare_only {
        None
    } else {
        open_truth(config.groundtruth.as_deref())?
    };
    let mut query_source = if config.prepare_only {
        None
    } else {
        Some(open_queries(config)?)
    };

    // Load the base vectors before constructing the index. Index loading takes
    // ownership of this allocation; the loader retains no second base copy.
    let mut data = LoadedBase::<B::Element>::load(config)?;
    let points = data.num_points;

    // Keep all search/index work inside a dedicated Rayon pool so the CLI's
    // requested worker count does not depend on the process-wide Rayon pool.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(config.sweep.threads)
        .build()
        .map_err(Error::thread_pool)?;

    let index = pool.install(|| B::load_index(config, &mut data))?;

    // `prepare_only` intentionally stops after all required index preparation
    // and loading work has completed.
    if config.prepare_only {
        return Ok(());
    }

    let query_source = query_source.as_mut().ok_or("missing query source")?;

    let gt =
        gt_file.map(|file| GroundTruthReader::new(BufReader::new(file), config.sweep.k, points));
    let writer = open_output(output)?;

    // The initial query batch serves two purposes:
    //   1. calibration;
    //   2. the first actual search batch.
    //
    // The queries are therefore retained and searched below rather than read
    // again after calibration.
    let mut decoded = Vec::new();
    query_source.read_batch(&mut decoded, config.sweep.calibration_samples)?;

    if decoded.is_empty() {
        return Err("query source is empty".into());
    }

    // Prepare sidecars and optional automatic residency before calibration.
    // Keep the guard alive through every search batch, just like Index does.
    let _residency = pool.install(|| B::prepare_search(&index, config))?;

    let calibration = pool
        .install(|| {
            index.calibrate(
                &decoded,
                config.sweep.calibration_l,
                config.window_size,
                orion_ann_core::CalibrationConfig {
                    k: config.sweep.k,
                    metric: config.metric.calibration_metric(),
                },
            )
        })
        .map_err(Error::from)?;

    let k = config.sweep.k;
    let summary = RunSummary {
        dataset: &config.dataset,
        k,
        batch_size: config.query_batch_size,
        residency: serde_json::to_value(_residency.report())?,
        calibration: serde_json::json!({
            "threshold":calibration.threshold,"early_exit_limit":calibration.early_exit_limit,
            "search_list_size":config.sweep.calibration_l,
        }),
    };
    stream_results(
        query_source.as_mut(),
        decoded,
        gt,
        writer,
        summary,
        |decoded| {
            pool.install(|| {
                B::search_batch(
                    &index,
                    decoded,
                    config,
                    config.sweep.k,
                    config.sweep.search_list_sizes[0],
                    calibration,
                )
            })
        },
    )
}

fn open_truth(path: Option<&std::path::Path>) -> Result<Option<std::fs::File>> {
    path.map(|p| std::fs::File::open(p).map_err(|e| Error::io("open ground truth", p, e)))
        .transpose()
}

fn open_output(output: Option<&PathBuf>) -> Result<Box<dyn Write>> {
    Ok(match output {
        Some(path) => Box::new(BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| format!("result output {}: {e}", path.display()))?,
        )),

        None => Box::new(BufWriter::new(std::io::stdout())),
    })
}

struct RunSummary<'a> {
    dataset: &'a str,
    k: usize,
    batch_size: usize,
    calibration: serde_json::Value,
    residency: serde_json::Value,
}

fn stream_results(
    source: &mut dyn QuerySource,
    mut decoded: Vec<f32>,
    mut gt: Option<GroundTruthReader<BufReader<std::fs::File>>>,
    mut writer: Box<dyn Write>,
    summary: RunSummary<'_>,
    search: impl Fn(&[f32]) -> Result<orion_ann::SearchResult>,
) -> Result<()> {
    use orion_ann_core::algorithm::search::take_search_counts;
    take_search_counts();
    let (mut count, mut recall_sum, mut search_seconds) = (0usize, 0.0, 0.0);

    // `stream_seconds` measures the complete streaming phase, including result
    // serialization, ground-truth evaluation, and query decoding.
    //
    // `search_seconds` below measures only execution of T::search.
    let stream_started = Instant::now();

    loop {
        let started = Instant::now();

        let results = search(&decoded)?;
        search_seconds += started.elapsed().as_secs_f64();
        for ids in results.rows() {
            // Ground truth is consumed in lockstep with the query stream.
            // This also detects ground-truth files that end too early.
            if let Some(truth) = gt.as_mut() {
                let row = truth
                    .next()?
                    .ok_or_else(|| format!("ground truth ended before query {count}"))?;

                recall_sum +=
                    ids.iter().filter(|id| row.contains(id)).count() as f64 / summary.k as f64;
            }

            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "type": "result",
                    "query_id": count,
                    "ids": ids,
                }),
            )
            .map_err(Error::from)?;

            writer.write_all(b"\n").map_err(Error::from)?;

            count += 1;
        }

        // Flush once per batch so streamed output remains externally observable
        // without paying the cost of flushing once per query.
        writer.flush().map_err(Error::from)?;

        // Refill the reusable decode buffer. A zero-length batch marks EOF.
        if source.read_batch(&mut decoded, summary.batch_size)? == 0 {
            break;
        }
    }

    // The opposite ground-truth length mismatch is checked only after all
    // queries have been consumed.
    if let Some(truth) = gt.as_mut() {
        if truth.next()?.is_some() {
            return Err("ground truth has more rows than the query stream".into());
        }
    }

    let (ndc_f32, ndc_byte, visits) = take_search_counts();
    let recall = gt.as_ref().map(|_| recall_sum / count as f64);

    println!(
        "{}",
        serde_json::json!({
            "type": "summary",
            "dataset": summary.dataset,
            "queries": count,
            "recall": recall,
            "k": summary.k,
            "calibration":summary.calibration,
            "residency":summary.residency,
            "search_seconds": search_seconds,
            "qps": count as f64 / search_seconds,
            "stream_seconds": stream_started.elapsed().as_secs_f64(),
            "ndc_f32": ndc_f32,
            "ndc_byte": ndc_byte,
            "visits": visits,
        })
    );

    Ok(())
}

fn run_snapshot(command: &Command, directory: &std::path::Path) -> Result<()> {
    let args = &command.run;
    let k = args.k.unwrap_or(10);
    let beam = command.search_l.unwrap_or(k.max(64));
    if args.query_batch_size == 0 || k == 0 || beam < k {
        return Err(Error::invalid_argument(
            "require positive query batch size and 1 <= k <= search L",
        ));
    }
    let base = args.base.as_ref().ok_or("--base is required")?;
    let query = args.query.as_deref().ok_or("--query is required")?;
    let index = orion_ann::OpenOptions {
        threads: args.threads,
        memory_budget_gib: args.memory_budget_gib,
        auto_pin: args.no_auto_pin.then_some(false),
    }
    .open(directory, orion_ann::BaseInput::from_file(base))?;
    let calibration = index.calibration(k).ok_or_else(|| {
        Error::invalid_state(format!("snapshot has no saved calibration for k={k}"))
    })?;
    // Snapshot open already applies the same automatic residency policy as Rust/Python.
    let summary = RunSummary {
        dataset: "snapshot",
        k,
        batch_size: args.query_batch_size,
        calibration: serde_json::to_value(calibration)?,
        residency: index.info()["residency"].clone(),
    };
    let format = match args.query_format {
        Some(format) => format,
        None => orion_ann::VectorFormat::infer(query)?,
    };
    let mut source = open_snapshot_queries(query, format, index.dimension(), index.storage())?;
    let gt = open_truth(args.groundtruth.as_deref())?
        .map(|f| GroundTruthReader::new(BufReader::new(f), k, index.len()));
    let mut decoded = Vec::new();
    if source.read_batch(&mut decoded, args.query_batch_size)? == 0 {
        return Err("query source is empty".into());
    }
    stream_results(
        source.as_mut(),
        decoded,
        gt,
        open_output(command.output.as_ref())?,
        summary,
        |batch| {
            index.search(
                orion_ann::QueryBatch::from_f32(batch, index.dimension())?,
                &orion_ann::SearchOptions::new(k, beam),
            )
        },
    )
}

fn execute(command: Command) -> Result<()> {
    if let Some(directory) = &command.snapshot {
        return run_snapshot(&command, directory);
    }
    let config = command.run.resolve_search(command.search_l)?;

    let resolved = serde_json::to_string_pretty(&config).map_err(Error::from)?;

    if command.run.print_config {
        println!("{resolved}");
        return Ok(());
    }

    // Preflight resolves resource requirements without starting the actual run.
    if command.run.preflight {
        let report = resources::preflight_report(&config)?;

        resources::print_report(&report, command.run.preflight_format)?;

        return resources::enforce_budget(&report);
    }

    log::info!("Run settings: {resolved}");

    // Dispatch only on resident coordinate type; dimensions stay runtime values.
    match config.storage() {
        orion_ann::VectorStorageKind::F32 => {
            run::<F32CascadeBackend>(&config, command.output.as_ref())
        }

        orion_ann::VectorStorageKind::U8 => {
            run::<NativeU8L2Backend>(&config, command.output.as_ref())
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let command = Command::parse();
    let format = command.error_format;
    if let Err(error) = execute(command) {
        match format {
            ErrorFormat::Text => log::error!("{error}"),
            ErrorFormat::Json => eprintln!(
                "{}",
                serde_json::to_string(&error.report())
                    .expect("error report contains only JSON values")
            ),
        }
        std::process::exit(2);
    }
}
