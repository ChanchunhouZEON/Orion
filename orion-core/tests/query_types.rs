/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_core::search::{
    cascade::stage::{NoPrefilter, rerank::NoRerank},
    query::QueryInput,
};
use orion_ann_core::{
    DynamicCascade, DynamicSearchConfig, Orion, PaddedDataset, PaddedLayout, ScoreMetric,
};
use orion_ann_vector::{L2F32Distance, L2U8Distance};

fn edges(n: usize) -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    (0..n)
        .map(|i| {
            (
                (0..n as u32).filter(|&j| j != i as u32).collect(),
                vec![],
                vec![],
            )
        })
        .collect()
}
fn expected(query: &[f64], row: &[f64], metric: ScoreMetric) -> f32 {
    let dot = query.iter().zip(row).map(|(a, b)| a * b).sum::<f64>();
    match metric {
        ScoreMetric::SquaredL2 => query
            .iter()
            .zip(row)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>() as f32,
        ScoreMetric::InnerProduct => dot as f32,
        ScoreMetric::CosineSimilarity => (dot
            / (query.iter().map(|x| x * x).sum::<f64>().sqrt()
                * row.iter().map(|x| x * x).sum::<f64>().sqrt()))
        .clamp(-1., 1.) as f32,
    }
}
#[test]
fn every_query_type_crosses_both_base_types_and_all_metrics() {
    let dim = 33;
    let bytes: Vec<u8> = (0..16 * dim).map(|i| ((i * 17) % 251 + 1) as u8).collect();
    let floats: Vec<f32> = bytes.iter().map(|&x| f32::from(x)).collect();
    let f = Orion::from_graph_partitions(
        PaddedDataset::from_packed(
            &floats,
            PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap(),
        )
        .unwrap(),
        &edges(16),
        0,
        15,
        0,
    )
    .unwrap();
    let b = Orion::from_graph_partitions(
        PaddedDataset::from_packed(
            &bytes,
            PaddedLayout::for_kernel::<L2U8Distance>(dim).unwrap(),
        )
        .unwrap(),
        &edges(16),
        0,
        15,
        0,
    )
    .unwrap();
    let qf: Vec<f32> = (0..dim).map(|i| i as f32 * 0.5 - 3.25).collect();
    let qb: Vec<u8> = (0..dim).map(|i| (i * 7) as u8).collect();
    let qi: Vec<i8> = (0..dim).map(|i| i as i8 - 17).collect();
    let q16: Vec<u16> = (0..dim).map(|i| 65_000 + i as u16).collect();
    let q32: Vec<u32> = (0..dim).map(|i| u32::MAX - i as u32 * 71).collect();
    let cases = [
        (
            QueryInput::F32(&qf),
            qf.iter().map(|&x| f64::from(x)).collect::<Vec<_>>(),
        ),
        (
            QueryInput::U8(&qb),
            qb.iter().map(|&x| f64::from(x)).collect(),
        ),
        (
            QueryInput::I8(&qi),
            qi.iter().map(|&x| f64::from(x)).collect(),
        ),
        (
            QueryInput::U16(&q16),
            q16.iter().map(|&x| f64::from(x)).collect(),
        ),
        (
            QueryInput::U32(&q32),
            q32.iter().map(|&x| f64::from(x)).collect(),
        ),
    ];
    for metric in [
        ScoreMetric::SquaredL2,
        ScoreMetric::InnerProduct,
        ScoreMetric::CosineSimilarity,
    ] {
        let ef = f.exact_stage(metric).unwrap();
        let eb = b.exact_stage(metric).unwrap();
        for (query, reference) in &cases {
            let config = DynamicSearchConfig {
                k: 16,
                search_list_size: 16,
                ..Default::default()
            };
            let recipe = |exact| DynamicCascade {
                prefilter: &NoPrefilter,
                admission: exact,
                rerank: &NoRerank,
            };
            for result in [
                f.search(*query, config, recipe(&ef), Some(metric)).unwrap(),
                b.search(*query, config, recipe(&eb), Some(metric)).unwrap(),
            ] {
                assert_eq!(result.counts, [16]);
                for (&id, &score) in result
                    .ids
                    .iter()
                    .zip(&result.scores.as_ref().unwrap().values)
                {
                    let row: Vec<f64> = floats[id as usize * dim..(id as usize + 1) * dim]
                        .iter()
                        .map(|&x| f64::from(x))
                        .collect();
                    let exact = expected(reference, &row, metric);
                    assert!(
                        (score - exact).abs() <= 3e-6 * exact.abs().max(1.),
                        "type={} metric={metric:?} id={id} {score} != {exact}",
                        query.element_type()
                    );
                }
            }
        }
    }
}

#[test]
fn u32_low_bits_survive_exact_rerank_and_subset_scoring() {
    let base = PaddedDataset::from_packed(
        &[16_777_216f32, 0., 16_777_220., 0.],
        PaddedLayout::for_kernel::<L2F32Distance>(2).unwrap(),
    )
    .unwrap();
    let index = Orion::from_graph_partitions(base, &edges(2), 0, 1, 0).unwrap();
    let exact = index.exact_stage(ScoreMetric::SquaredL2).unwrap();
    let q = QueryInput::U32(&[16_777_217, 0]);
    let result = index
        .search(
            q,
            DynamicSearchConfig {
                k: 2,
                search_list_size: 2,
                ..Default::default()
            },
            DynamicCascade {
                prefilter: &NoPrefilter,
                admission: &exact,
                rerank: &exact,
            },
            Some(ScoreMetric::SquaredL2),
        )
        .unwrap();
    assert_eq!(result.ids, [0, 1]);
    assert_eq!(result.scores.as_ref().unwrap().values, [1., 9.]);
    let filtered = index
        .search_subset_queries(q, &[0, 1], 2, ScoreMetric::SquaredL2, true)
        .unwrap();
    assert_eq!(result, filtered);
}
