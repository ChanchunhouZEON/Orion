/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! File-based application round trip, including business IDs and exact scores.
//! Run with --features file-queries and arguments:
//! BASE QUERY IDS_JSON OPTIONS_JSON SNAPSHOT_DIR
//! IDS_JSON is the application's unique u64 ID array in base-row order.
use orion_ann::{BaseInput, Index, IndexOptions, QueryBatch, SearchOptions};
use std::{fs::File, io::BufReader, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err("usage: business_search BASE QUERY IDS_JSON OPTIONS_JSON SNAPSHOT_DIR".into());
    }
    let ids = serde_json::from_reader(BufReader::new(File::open(&args[2])?))?;
    let options: IndexOptions = serde_json::from_reader(BufReader::new(File::open(&args[3])?))?;
    let mut index = Index::new_with_ids(BaseInput::from_file(&args[0]), options, ids)?;
    let queries = orion_ann::read_query_file(
        Path::new(&args[1]),
        None,
        index.dimension(),
        index.storage(),
    )?;
    let warmup = queries[..queries.len().min(200 * index.dimension())].to_vec();
    index.calibrate_many(warmup, &[10, 100], None)?;
    let controls = SearchOptions::new(10, 64).with_scores(true);
    // Keep this demonstration's response small; full batch searches use the same API.
    let queries = &queries[..queries.len().min(4 * index.dimension())];
    let result = index.search(QueryBatch::from_f32(queries, index.dimension())?, &controls)?;
    let reference = index.save(Path::new(&args[4]))?;
    let reopened = Index::open(&reference, BaseInput::from_file(&args[0]))?;
    let restored = reopened.search(
        QueryBatch::from_f32(queries, reopened.dimension())?,
        &controls,
    )?;
    assert_eq!(result, restored);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "snapshot": reference,
            "calibrated_ks": reopened.calibrated_ks().collect::<Vec<_>>(),
            "counts": result.counts,
            "internal_ids": result.ids,
            "external_ids": result.external_ids,
            "score_metric": result.scores.as_ref().unwrap().metric.as_str(),
            "scores": result.scores.as_ref().unwrap().values,
        }))?
    );
    Ok(())
}
