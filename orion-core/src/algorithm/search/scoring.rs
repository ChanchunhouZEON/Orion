/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Exact scoring of selected results when no matching rerank score is available.
use crate::ScoreMetric;

/// Only the final returned candidates use this fallback. Accumulate in f64 to
/// avoid intermediate overflow; reject unrepresentable f32 outputs at assembly.
pub(crate) fn exact_score(query: &[f32], row: &[f32], metric: ScoreMetric) -> f32 {
    match metric {
        ScoreMetric::SquaredL2 => query
            .iter()
            .zip(row)
            .map(|(&q, &x)| {
                let d = f64::from(q) - f64::from(x);
                d * d
            })
            .sum::<f64>() as f32,
        ScoreMetric::InnerProduct => query
            .iter()
            .zip(row)
            .map(|(&q, &x)| f64::from(q) * f64::from(x))
            .sum::<f64>() as f32,
        ScoreMetric::CosineSimilarity => {
            let (mut dot, mut q2, mut x2) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (&q, &x) in query.iter().zip(row) {
                let (q, x) = (f64::from(q), f64::from(x));
                dot += q * x;
                q2 += q * q;
                x2 += x * x;
            }
            // Zero-vector cosine is undefined. Input normalization rejects it.
            (dot / (q2.sqrt() * x2.sqrt())).clamp(-1.0, 1.0) as f32
        }
    }
}
