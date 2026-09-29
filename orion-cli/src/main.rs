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
use orion_cli::{
    cli::{
        config::{Args, ResolvedRunConfig},
        data::LoadedBase,
        execution::SearchBackend,
        plan::SearchPlan,
        query::{open_queries, GroundTruthReader},
        resources,
    },
    with_supported_dimension,
};
use std::{
    io::{BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use orion_cli::cli::execution::{F32CascadeBackend, NativeU8L2Backend};

#[derive(Parser)]
#[command(
    name = "orion",
    about = "Build or search an Orion index; ground truth is optional"
)]
struct Command {
    #[command(flatten)]
    run: Args,

    /// Single beam width; defaults to max(k, 64).
    /// Sweeps use `orion-sweep`.
    #[arg(long, conflicts_with = "search_list_sizes")]
    search_l: Option<usize>,

    /// Write result IDs as JSONL to a new file.
    /// Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

fn run<const N: usize, B: SearchBackend>(
    config: &ResolvedRunConfig,
    output: Option<&PathBuf>,
) -> Result<(), String>
where
    [B::Element; N]:
    vector::FullPrecisionDistance<B::Element, N>,
{
    // Open lightweight query/evaluation inputs before allocating the potentially
    // much larger base dataset and index structures.
    let gt_file = if config.prepare_only {
        None
    } else {
        config
            .groundtruth
            .as_ref()
            .map(|path| {
                std::fs::File::open(path)
                    .map_err(|e| format!("ground truth {}: {e}", path.display()))
            })
            .transpose()?
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
        .map_err(|e| e.to_string())?;

    let index = pool.install(|| B::load_index::<N>(config, &mut data))?;

    // `prepare_only` intentionally stops after all required index preparation
    // and loading work has completed.
    if config.prepare_only {
        return Ok(());
    }

    let query_source = query_source
        .as_mut()
        .ok_or("missing query source")?;

    let mut gt = gt_file.map(|file| {
        GroundTruthReader::new(
            BufReader::new(file),
            config.sweep.k,
            points,
        )
    });

    // Results are emitted as JSONL. `create_new` prevents silently overwriting
    // an existing result file.
    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .map_err(|e| {
                    format!("result output {}: {e}", path.display())
                })?,
        )),

        None => Box::new(BufWriter::new(std::io::stdout())),
    };

    // The initial query batch serves two purposes:
    //   1. calibration;
    //   2. the first actual search batch.
    //
    // The queries are therefore retained and searched below rather than read
    // again after calibration.
    let mut decoded = Vec::new();
    query_source.read_batch(
        &mut decoded,
        config.sweep.calibration_samples,
    )?;

    if decoded.is_empty() {
        return Err("query source is empty".into());
    }

    let mut queries: Vec<[f32; N]> = decoded
        .chunks_exact(N)
        .map(|query| query.try_into().unwrap())
        .collect();

    let calibration = pool
        .install(|| {
            index.calibrate(
                &queries,
                config.sweep.calibration_l,
                config.window_size,
                orion::CalibrationConfig {
                    k: config.sweep.k,
                    metric: config.metric.calibration_metric(),
                },
            )
        })
        .map_err(|e| e.to_string())?;

    // Apply implementation-specific pinning only after calibration so the
    // measured search phase reflects the final runtime configuration.
    B::pin_search_storage(&index, config);

    use orion::algorithm::search::{
        NDC_F32,
        NDC_I8,
        VISIT_COUNT,
    };

    // Reset global search counters immediately before the measured search phase.
    NDC_F32.reset();
    NDC_I8.reset();
    VISIT_COUNT.reset();

    let (mut count, mut recall_sum, mut search_seconds) =
        (0usize, 0.0, 0.0);

    // `stream_seconds` measures the complete streaming phase, including result
    // serialization, ground-truth evaluation, and query decoding.
    //
    // `search_seconds` below measures only execution of T::search.
    let stream_started = Instant::now();

    loop {
        // Calibration queries are searched first. Because they are the original
        // first queries from the input stream, query IDs remain contiguous and
        // preserve their source ordering.
        let started = Instant::now();

        let results = pool
            .install(|| {
                B::search_batch(
                    &index,
                    &queries,
                    config,
                    config.sweep.search_list_sizes[0],
                    calibration,
                )
            })
            .map_err(|e| e.to_string())?;

        search_seconds += started.elapsed().as_secs_f64();

        for ids in results {
            // Ground truth is consumed in lockstep with the query stream.
            // This also detects ground-truth files that end too early.
            if let Some(truth) = gt.as_mut() {
                let row = truth
                    .next()?
                    .ok_or_else(|| {
                        format!(
                            "ground truth ended before query {count}"
                        )
                    })?;

                recall_sum += ids
                    .iter()
                    .filter(|id| row.contains(id))
                    .count() as f64
                    / config.sweep.k as f64;
            }

            serde_json::to_writer(
                &mut writer,
                &serde_json::json!({
                    "type": "result",
                    "query_id": count,
                    "ids": ids,
                }),
            )
                .map_err(|e| e.to_string())?;

            writer
                .write_all(b"\n")
                .map_err(|e| e.to_string())?;

            count += 1;
        }

        // Flush once per batch so streamed output remains externally observable
        // without paying the cost of flushing once per query.
        writer
            .flush()
            .map_err(|e| e.to_string())?;

        // Refill the reusable decode buffer. A zero-length batch marks EOF.
        if query_source.read_batch(
            &mut decoded,
            config.query_batch_size,
        )? == 0
        {
            break;
        }

        queries.clear();
        queries.extend(
            decoded
                .chunks_exact(N)
                .map(|query| <[f32; N]>::try_from(query).unwrap()),
        );
    }

    // The opposite ground-truth length mismatch is checked only after all
    // queries have been consumed.
    if let Some(truth) = gt.as_mut() {
        if truth.next()?.is_some() {
            return Err(
                "ground truth has more rows than the query stream".into(),
            );
        }
    }

    let recall = gt
        .as_ref()
        .map(|_| recall_sum / count as f64);

    println!(
        "{}",
        serde_json::json!({
            "type": "summary",
            "dataset": config.dataset,
            "queries": count,
            "recall": recall,
            "k": config.sweep.k,
            "search_seconds": search_seconds,
            "qps": count as f64 / search_seconds,
            "stream_seconds": stream_started.elapsed().as_secs_f64(),
            "ndc_f32": NDC_F32.drain(),
            "ndc_byte": NDC_I8.drain(),
            "visits": VISIT_COUNT.drain(),
        })
    );

    Ok(())
}

fn execute(mut command: Command) -> Result<(), String> {
    // `--search-l` is the single-run shorthand for supplying exactly one search
    // list size through the shared sweep configuration.
    if let Some(search_l) = command.search_l {
        command.run.sweep.search_list_sizes = Some(vec![search_l]);
    }

    let config = command.run.resolve_search()?;

    // The search implementation uses a statically bounded worker structure,
    // therefore reject unsupported thread counts before allocating resources.
    if config.sweep.threads > orion::algorithm::search::MAX_WORKERS {
        return Err(format!(
            "--threads exceeds MAX_WORKERS={}",
            orion::algorithm::search::MAX_WORKERS,
        ));
    }

    let resolved = serde_json::to_string_pretty(&config)
        .map_err(|e| e.to_string())?;

    if command.run.print_config {
        println!("{resolved}");
        return Ok(());
    }

    log::info!("Run settings: {resolved}");

    // Preflight resolves resource requirements without starting the actual run.
    if command.run.preflight {
        let report = resources::preflight_report(&config)?;

        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|e| e.to_string())?
        );

        log::info!("{}", resources::format_report(&report));

        return resources::enforce_budget(&report);
    }

    // Dispatch to the concrete vector representation and compile-time
    // dimensionality selected by the resolved search plan.
    match config.search_plan() {
        SearchPlan::F32Cascade { .. } => {
            with_supported_dimension!(
                config.dimension,
                |D| run::<D, F32CascadeBackend>(
                    &config,
                    command.output.as_ref(),
                )
            )
        }

        SearchPlan::NativeU8L2 => {
            run::<128, NativeU8L2Backend>(
                &config,
                command.output.as_ref(),
            )
        }
    }
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
        .init();

    if let Err(error) = execute(Command::parse()) {
        log::error!("{error}");
        std::process::exit(2);
    }
}