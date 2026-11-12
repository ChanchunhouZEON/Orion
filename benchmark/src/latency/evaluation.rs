/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Untimed parity and strict ID-recall evaluation. Ground truth is never touched
//! by timed clients; every unique query is evaluated exactly once here.
use anyhow::{ensure, Result};
use orion_ann::{Index, QueryBatch, SearchOptions, SearchResult};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize)]
pub struct Evaluation {
    pub unique_queries: usize,
    pub recall: Option<f64>,
    pub partial_results: usize,
    pub ids_sha256: String,
    pub batch_parity: bool,
}

pub fn validate_truth(truth: &[Vec<u32>], queries: usize, k: usize) -> Result<()> {
    ensure!(truth.len() == queries, "ground truth/query count mismatch");
    ensure!(
        truth
            .iter()
            .all(|r| r.len() == k && r.iter().collect::<BTreeSet<_>>().len() == k),
        "ground truth top-k rows must contain k distinct IDs"
    );
    Ok(())
}

pub fn recall_row(ids: &[u32], truth: &[u32], k: usize) -> Result<f64> {
    ensure!(
        k > 0 && ids.len() <= k && truth.len() == k,
        "invalid recall row shape"
    );
    let ids: BTreeSet<_> = ids.iter().copied().collect();
    let truth: BTreeSet<_> = truth.iter().copied().collect();
    ensure!(truth.len() == k, "duplicate ground truth ID");
    Ok(ids.intersection(&truth).count() as f64 / k as f64)
}

/// Verify requests at this concurrency against a batch reference. Compare only
/// valid score prefixes, since padding is deliberately NaN.
pub fn evaluate(
    index: &Index,
    rows: &[QueryBatch<'_>],
    options: &SearchOptions,
    reference: &SearchResult,
    truth: Option<&[Vec<u32>]>,
    concurrency: usize,
) -> Result<Evaluation> {
    ensure!(
        (1..=64).contains(&concurrency) && !rows.is_empty(),
        "invalid evaluation workload"
    );
    ensure!(
        reference.k == options.k && reference.len() == rows.len(),
        "reference shape mismatch"
    );
    if let Some(truth) = truth {
        validate_truth(truth, rows.len(), options.k)?;
    }
    let parts = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..concurrency)
            .map(|worker| {
                scope.spawn(move || -> Result<_> {
                    let mut answers = Vec::new();
                    for row in (worker..rows.len()).step_by(concurrency) {
                        answers.push((row, index.search(rows[row], options)?));
                    }
                    Ok(answers)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("evaluation client panicked"))
                    .and_then(|r| r)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    let mut answers: Vec<_> = parts.into_iter().flatten().collect();
    answers.sort_by_key(|(row, _)| *row);
    let mut hits = 0.0;
    let mut partial = 0;
    let mut digest = Sha256::new();
    for (row, answer) in answers {
        let n = reference.counts[row] as usize;
        let offset = row * options.k;
        ensure!(
            answer.counts == [n as u32] && answer.ids == reference.ids[offset..offset + options.k],
            "single/batch result mismatch at query {row}"
        );
        match (&answer.scores, &reference.scores) {
            (Some(a), Some(b)) => ensure!(
                a.metric == b.metric && a.values[..n] == b.values[offset..offset + n],
                "score mismatch at query {row}"
            ),
            (None, None) => (),
            _ => anyhow::bail!("score presence mismatch"),
        }
        match (&answer.external_ids, &reference.external_ids) {
            (Some(a), Some(b)) => ensure!(a[..n] == b[offset..offset + n], "external ID mismatch"),
            (None, None) => (),
            _ => anyhow::bail!("external ID presence mismatch"),
        }
        digest.update((n as u32).to_le_bytes());
        for id in &answer.ids[..n] {
            digest.update(id.to_le_bytes());
        }
        partial += usize::from(n < options.k);
        if let Some(truth) = truth {
            hits += recall_row(&answer.ids[..n], &truth[row], options.k)?;
        }
    }
    Ok(Evaluation {
        unique_queries: rows.len(),
        recall: truth.map(|_| hits / rows.len() as f64),
        partial_results: partial,
        ids_sha256: format!("{:x}", digest.finalize()),
        batch_parity: true,
    })
}
