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
    GroundTruthReader, QuerySource, ResolvedRunConfig, open_queries, open_snapshot_queries,
    resources,
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
        long, value_name = "DIR", requires = "query",
        conflicts_with_all = [
            "cache_dir", "staged_file",
            "graph_source", "graph_degree", "build_l", "alpha", "max_extra",
            "max_points", "window_size", "base_format", "vector_storage", "dimension",
            "metric", "prefilter", "admission", "rerank", "print_config", "preflight",
            "prepare_only"
        ]
    )]
    #[cfg_attr(
        feature = "experimental-graph",
        arg(conflicts_with = "supplier_staged")
    )]
    snapshot: Option<PathBuf>,

    /// Relocate snapshot objects; all payload hashes are still verified.
    #[arg(long, requires = "snapshot")]
    object_store: Option<PathBuf>,

    /// Single beam width; defaults to max(k, 64).
    /// Sweeps use `orion-sweep`.
    #[arg(long, alias = "ls", alias = "search-list-sizes")]
    search_l: Option<usize>,

    /// Write result IDs as JSONL to a new file.
    /// Defaults to stdout.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Include exact original-metric scores of returned candidates.
    #[arg(long)]
    scores: bool,

    /// JSON array of unique u64 business IDs, one per indexed row.
    #[arg(long, conflicts_with = "snapshot")]
    #[cfg_attr(
        feature = "experimental-graph",
        arg(conflicts_with = "supplier_staged")
    )]
    external_ids: Option<PathBuf>,

    /// JSON LabelFilter for exact subset search of a saved label revision.
    #[arg(long, requires = "snapshot", conflicts_with = "search_l")]
    label_filter: Option<PathBuf>,

    /// Maximum matching rows evaluated by the filtered exact baseline.
    #[arg(long, requires = "label_filter")]
    filter_max_candidates: Option<usize>,

    /// Structured errors on stderr for services; success output is unchanged.
    #[arg(long, value_enum, default_value = "text")]
    error_format: ErrorFormat,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum ErrorFormat {
    Text,
    Json,
}

fn run(config: &ResolvedRunConfig, command: &Command) -> Result<()> {
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

    let input = orion_ann::BaseInput::from_file(&config.base);
    let options = command.run.index_options();
    #[cfg(feature = "experimental-graph")]
    let supplied = command
        .run
        .supplier_staged
        .as_ref()
        .map(|path| {
            orion_ann::Index::from_staged(
                orion_ann::BaseInput::from_file(&config.base),
                path,
                options.clone(),
            )
        })
        .transpose()?;
    #[cfg(not(feature = "experimental-graph"))]
    let supplied: Option<orion_ann::Index> = None;
    let mut index = if let Some(index) = supplied {
        index
    } else if let Some(path) = &command.external_ids {
        let ids: Vec<u64> = serde_json::from_reader(BufReader::new(
            std::fs::File::open(path).map_err(|e| Error::io("open external IDs", path, e))?,
        ))?;
        orion_ann::Index::new_with_ids(input, options, ids)?
    } else {
        orion_ann::Index::new(input, options)?
    };
    let points = index.len();
    log::info!("Prepared index: {}", index.info());

    // `prepare_only` intentionally stops after all required index preparation
    // and loading work has completed.
    if config.prepare_only {
        return Ok(());
    }

    let query_source = query_source.as_mut().ok_or("missing query source")?;

    let gt =
        gt_file.map(|file| GroundTruthReader::new(BufReader::new(file), config.sweep.k, points));
    let writer = open_output(command.output.as_ref())?;

    // The initial query batch serves two purposes:
    //   1. calibration;
    //   2. the first actual search batch.
    //
    // The queries are therefore retained and searched below rather than read
    // again after calibration.
    let mut decoded = orion_ann::QueryBuffer::default();
    query_source.read_typed_batch(&mut decoded, config.sweep.calibration_samples)?;

    if decoded.is_empty() {
        return Err("query source is empty".into());
    }

    let calibration = index.calibrate_batch(
        orion_ann::QueryBatch::from_input(decoded.as_input(), index.dimension())?,
        config.sweep.k,
        Some(config.sweep.calibration_l),
    )?;

    let k = config.sweep.k;
    let summary = RunSummary {
        dataset: &config.dataset,
        label_revision: None,
        k,
        batch_size: config.query_batch_size,
        residency: index.info()["residency"].clone(),
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
            index.search(
                orion_ann::QueryBatch::from_input(decoded, index.dimension())?,
                &orion_ann::SearchOptions::new(config.sweep.k, config.sweep.search_list_sizes[0])
                    .with_scores(command.scores),
            )
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
    label_revision: Option<u64>,
    residency: serde_json::Value,
}

fn stream_results(
    source: &mut dyn QuerySource,
    mut decoded: orion_ann::QueryBuffer,
    mut gt: Option<GroundTruthReader<BufReader<std::fs::File>>>,
    mut writer: Box<dyn Write>,
    summary: RunSummary<'_>,
    search: impl Fn(orion_ann::QueryInput<'_>) -> Result<orion_ann::SearchResult>,
) -> Result<()> {
    use orion_ann_core::search::take_search_counts;
    take_search_counts();
    let (mut count, mut recall_sum, mut search_seconds) = (0usize, 0.0, 0.0);

    // `stream_seconds` measures the complete streaming phase, including result
    // serialization, ground-truth evaluation, and query decoding.
    //
    // `search_seconds` below measures only execution of T::search.
    let stream_started = Instant::now();

    loop {
        let started = Instant::now();

        let results = search(decoded.as_input())?;
        search_seconds += started.elapsed().as_secs_f64();
        for (row_index, ids) in results.rows().enumerate() {
            // Ground truth is consumed in lockstep with the query stream.
            // This also detects ground-truth files that end too early.
            if let Some(truth) = gt.as_mut() {
                let row = truth
                    .next()?
                    .ok_or_else(|| format!("ground truth ended before query {count}"))?;

                recall_sum +=
                    ids.iter().filter(|id| row.contains(id)).count() as f64 / summary.k as f64;
            }

            let mut record = serde_json::json!({ "type": "result", "query_id": count, "ids": ids });
            if let Some(revision) = summary.label_revision {
                record["label_revision"] = revision.into();
            }
            if let Some(scores) = &results.scores {
                let start = row_index * results.k;
                record["scores"] = serde_json::json!(&scores.values[start..start + ids.len()]);
                record["score_metric"] = scores.metric.as_str().into();
            }
            if let Some(external) = &results.external_ids {
                let start = row_index * results.k;
                record["external_ids"] = serde_json::json!(&external[start..start + ids.len()]);
            }
            serde_json::to_writer(&mut writer, &record).map_err(Error::from)?;

            writer.write_all(b"\n").map_err(Error::from)?;

            count += 1;
        }

        // Flush once per batch so streamed output remains externally observable
        // without paying the cost of flushing once per query.
        writer.flush().map_err(Error::from)?;

        // Refill the reusable decode buffer. A zero-length batch marks EOF.
        if source.read_typed_batch(&mut decoded, summary.batch_size)? == 0 {
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
            "label_revision": summary.label_revision,
            "search_mode": if summary.label_revision.is_some() { "filtered_exact" } else { "adaptive_ann" },
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
    let query = args.query.as_deref().ok_or("--query is required")?;
    let options = orion_ann::OpenOptions {
        object_store: command.object_store.clone(),
        threads: args.threads,
        memory_budget_gib: args.memory_budget_gib,
        auto_pin: args.no_auto_pin.then_some(false),
        storage_options: (args.storage_options.is_some() || args.mmap_dir.is_some()).then(|| {
            orion_ann::StorageOptions {
                mode: args.storage_options.unwrap_or_default(),
                directory: args.mmap_dir.clone(),
            }
        }),
    };
    let index = match &args.base {
        Some(base) => options.open(directory, orion_ann::BaseInput::from_file(base))?,
        None => options.open_bundle(directory)?,
    };
    let filter: Option<orion_ann::LabelFilter> = command
        .label_filter
        .as_ref()
        .map(|path| {
            use std::io::Read;
            let file =
                std::fs::File::open(path).map_err(|e| Error::io("open label filter", path, e))?;
            let mut bytes = Vec::new();
            file.take(1024 * 1024 + 1).read_to_end(&mut bytes)?;
            if bytes.len() > 1024 * 1024 {
                return Err(Error::invalid_argument("label filter JSON exceeds 1 MiB"));
            }
            serde_json::from_slice(&bytes)
                .map_err(|e| Error::invalid_argument(format!("invalid label filter JSON: {e}")))
        })
        .transpose()?;
    let calibration = if filter.is_some() {
        serde_json::Value::Null
    } else {
        serde_json::to_value(index.calibration(k).ok_or_else(|| {
            Error::invalid_state(format!("snapshot has no saved calibration for k={k}"))
        })?)?
    };
    let summary = RunSummary {
        dataset: "snapshot",
        k,
        batch_size: args.query_batch_size,
        calibration,
        label_revision: if filter.is_some() {
            Some(index.label_revision()?)
        } else {
            None
        },
        residency: index.info()["residency"].clone(),
    };
    let format = match args.query_format {
        Some(format) => format,
        None => orion_ann::VectorFormat::infer(query)?,
    };
    let mut source = open_snapshot_queries(query, format, index.dimension(), index.storage())?;
    let gt = open_truth(args.groundtruth.as_deref())?
        .map(|f| GroundTruthReader::new(BufReader::new(f), k, index.len()));
    let mut decoded = orion_ann::QueryBuffer::default();
    if source.read_typed_batch(&mut decoded, args.query_batch_size)? == 0 {
        return Err("query source is empty".into());
    }
    stream_results(
        source.as_mut(),
        decoded,
        gt,
        open_output(command.output.as_ref())?,
        summary,
        |batch| {
            if let Some(filter) = &filter {
                return Ok(index
                    .search_filtered(
                        orion_ann::QueryBatch::from_input(batch, index.dimension())?,
                        filter,
                        &orion_ann::FilterSearchOptions {
                            k,
                            scores: command.scores,
                            max_candidates: command.filter_max_candidates.unwrap_or(100_000),
                        },
                    )?
                    .result);
            }
            index.search(
                orion_ann::QueryBatch::from_input(batch, index.dimension())?,
                &orion_ann::SearchOptions::new(k, beam).with_scores(command.scores),
            )
        },
    )
}

fn execute(command: Command) -> Result<()> {
    if let Some(directory) = &command.snapshot {
        return run_snapshot(&command, directory);
    }
    let config = if command.run.preflight || command.run.print_config {
        command.run.resolve_search(command.search_l)?
    } else {
        command.run.resolve_execution(command.search_l)?
    };

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

    log::info!("Input settings (graph identity finalized during load): {resolved}");

    run(&config, &command)
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
