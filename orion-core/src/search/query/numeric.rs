/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Wide reference arithmetic for mixed coordinates. No base copy or per-ID
//! allocation; existing homogeneous SIMD kernels remain the preferred path.
use crate::ScoreMetric;
pub(crate) fn score<Q: Copy + Into<f64>, T: Copy + Into<f32>>(
    query: &[Q],
    row: &[T],
    metric: ScoreMetric,
) -> f32 {
    let (mut squared, mut dot, mut nq, mut nb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&q, &b) in query.iter().zip(row) {
        let (q, b) = (q.into(), f64::from(b.into()));
        let delta = q - b;
        squared += delta * delta;
        dot += q * b;
        nq += q * q;
        nb += b * b;
    }
    match metric {
        ScoreMetric::SquaredL2 => squared as f32,
        ScoreMetric::InnerProduct => dot as f32,
        ScoreMetric::CosineSimilarity => (dot / (nq.sqrt() * nb.sqrt())).clamp(-1.0, 1.0) as f32,
    }
}
