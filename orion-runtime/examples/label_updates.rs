/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! cargo run -p orion-ann --example label_updates -- /tmp/orion-label-demo
use orion_ann::{
    BaseInput, ExternalLabelUpdate, FilterSearchOptions, Index, IndexOptions, LabelFilter,
    LabelKey, QueryBatch, Result,
};
fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(std::path::PathBuf::from)
        .ok_or_else(|| orion_ann::Error::invalid_argument("pass a new demo output directory"))?;
    let base: Vec<f32> = (0..64).flat_map(|i| [i as f32 + 1., 1., 2.]).collect();
    let index = Index::new_with_ids(
        BaseInput::from_f32(&base, 3)?,
        IndexOptions {
            cache_dir: root.join("cache"),
            threads: 2,
            auto_pin: false,
            ..Default::default()
        },
        (2000..2064).collect(),
    )?;
    let book = LabelKey::new("category", "book")?;
    let movie = LabelKey::new("category", "movie")?;
    let business_ids = vec![2000, 2001, 2002];
    let revision = index.update_labels_external(
        0,
        &[ExternalLabelUpdate {
            label: book.clone(),
            add: business_ids.clone(),
            remove: vec![],
        }],
    )?;
    let filter = LabelFilter {
        all: vec![book.clone()],
        ..Default::default()
    };
    let query = QueryBatch::from_f32(&[1., 1., 2.], 3)?;
    let options = FilterSearchOptions::new(3).with_scores(true);
    let before = index.search_filtered(query, &filter, &options)?;
    let saved = index.save(&root.join("revision-1"))?;
    let revision = index.update_labels_external(
        revision,
        &[
            ExternalLabelUpdate {
                label: book,
                add: vec![],
                remove: business_ids.clone(),
            },
            ExternalLabelUpdate {
                label: movie,
                add: business_ids,
                remove: vec![],
            },
        ],
    )?;
    let after = index.search_filtered(query, &filter, &options)?;
    assert_eq!(after.result.counts, [0]);
    let restored = Index::open(&saved, BaseInput::from_f32(&base, 3)?)?;
    assert_eq!(restored.search_filtered(query, &filter, &options)?, before);
    println!(
        "{}",
        serde_json::json!({
            "saved_revision":before.label_revision,"live_revision":revision,
            "before_external_ids":before.result.external_ids,
            "after_book_counts":after.result.counts,"snapshot":saved.path(),
            "restored_revision":restored.label_revision()?,"restored_results_match":true
        })
    );
    Ok(())
}
