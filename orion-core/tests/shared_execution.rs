/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Representation boundaries must not create a second traversal or result
//! contract. Exercise typed/float transport, observers, scores and failures on
//! the same graph, including dimensions that require padding.
use orion_ann_core::DynamicCascade;
use orion_ann_core::search::cascade::stage::rerank::NoRerank;
use orion_ann_core::search::diagnostics::NeighborMode;
use orion_ann_core::search::diagnostics::SearchObserver;
use orion_ann_core::search::query::QueryInput;
use orion_ann_core::{DynamicSearchConfig, Orion, PaddedDataset, ScoreMetric};
use orion_ann_vector::{IpU8Distance, L2F32Distance, PaddedLayout};

#[derive(Default, Debug, PartialEq)]
struct Trace(Vec<(u32, bool)>);
impl SearchObserver for Trace {
    fn expansion_started(&mut self, id: u32, converged: bool, _peel: bool) {
        self.0.push((id, converged));
    }
}
fn base(dim: usize) -> Vec<u8> {
    (0..24 * dim)
        .map(|i| ((i * 13 + i / dim * 19) % 29 + 1) as u8)
        .collect()
}
fn parts() -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>)> {
    (0..24u32)
        .map(|id| {
            let neighbors: Vec<_> = (1..12).map(|j| (id + j) % 24).collect();
            (
                neighbors[..5].to_vec(),
                neighbors[5..9].to_vec(),
                neighbors[9..].to_vec(),
            )
        })
        .collect()
}
fn index(dim: usize) -> Orion<u8> {
    Orion::from_graph_partitions(
        PaddedDataset::from_packed(
            &base(dim),
            PaddedLayout::for_kernel::<IpU8Distance>(dim).unwrap(),
        )
        .unwrap(),
        &parts(),
        0,
        12,
        2,
    )
    .unwrap()
}
fn controls() -> DynamicSearchConfig {
    DynamicSearchConfig {
        k: 5,
        search_list_size: 16,
        ..Default::default()
    }
}

#[test]
fn identity_metrics_share_transport_observer_and_score_contracts() {
    for dim in [1, 25, 128, 129] {
        let index = index(dim);
        let bytes = base(dim)[..3 * dim].to_vec();
        let floats: Vec<_> = bytes.iter().map(|&x| f32::from(x)).collect();
        for metric in [
            ScoreMetric::SquaredL2,
            ScoreMetric::InnerProduct,
            ScoreMetric::CosineSimilarity,
        ] {
            let exact = index.exact_stage(metric).unwrap();
            let recipe = DynamicCascade {
                prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
                admission: &exact,
                rerank: &NoRerank,
            };
            let ids = index
                .search_batch(QueryInput::U8(&bytes), controls(), recipe, None)
                .unwrap();
            let scored = index
                .search_batch(QueryInput::U8(&bytes), controls(), recipe, Some(metric))
                .unwrap();
            let transported = index
                .search_batch(QueryInput::F32(&floats), controls(), recipe, Some(metric))
                .unwrap();
            assert_eq!(ids.ids, scored.ids);
            assert_eq!(ids.counts, scored.counts);
            assert_eq!(scored, transported, "dim={dim}, metric={metric:?}");
            for (q, row) in bytes
                .chunks_exact(dim)
                .zip(scored.ids.chunks_exact(controls().k))
            {
                let mut trace = Trace::default();
                assert_eq!(
                    index
                        .search_observed(
                            QueryInput::U8(q),
                            controls(),
                            recipe,
                            NeighborMode::default(),
                            &mut trace
                        )
                        .unwrap(),
                    row
                );
                assert!(!trace.0.is_empty());
            }
            let empty = index
                .search_batch(QueryInput::U8(&[]), controls(), recipe, Some(metric))
                .unwrap();
            assert!(empty.ids.is_empty());
        }
    }
}

#[test]
fn byte_and_float_identity_l2_preserve_the_same_graph_walk() {
    let dim = 25;
    let bytes = index(dim);
    let values: Vec<_> = base(dim).iter().map(|&x| f32::from(x)).collect();
    let floats: Orion = Orion::from_graph_partitions(
        PaddedDataset::from_packed(
            &values,
            PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap(),
        )
        .unwrap(),
        &parts(),
        0,
        12,
        2,
    )
    .unwrap();
    let mut byte_trace = Trace::default();
    let mut float_trace = Trace::default();
    let byte_ids = bytes
        .search_observed(
            QueryInput::U8(&base(dim)[..dim]),
            controls(),
            DynamicCascade {
                prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
                admission: &bytes.exact_stage(ScoreMetric::SquaredL2).unwrap(),
                rerank: &NoRerank,
            },
            NeighborMode::default(),
            &mut byte_trace,
        )
        .unwrap();
    let float_ids = floats
        .search_observed(
            QueryInput::F32(&values[..dim]),
            controls(),
            DynamicCascade {
                prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
                admission: &floats.exact_stage(ScoreMetric::SquaredL2).unwrap(),
                rerank: &NoRerank,
            },
            NeighborMode::default(),
            &mut float_trace,
        )
        .unwrap();
    assert_eq!(byte_ids, float_ids);
    assert_eq!(byte_trace, float_trace);
    assert_eq!(
        bytes
            .search_batch(
                QueryInput::U8(&base(dim)),
                controls(),
                DynamicCascade {
                    prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
                    admission: &bytes.exact_stage(ScoreMetric::SquaredL2).unwrap(),
                    rerank: &NoRerank
                },
                Some(ScoreMetric::SquaredL2)
            )
            .unwrap(),
        floats
            .search_batch(
                QueryInput::F32(&values),
                controls(),
                DynamicCascade {
                    prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
                    admission: &floats.exact_stage(ScoreMetric::SquaredL2).unwrap(),
                    rerank: &NoRerank
                },
                Some(ScoreMetric::SquaredL2)
            )
            .unwrap()
    );
}

#[test]
fn rejected_rows_do_not_poison_prepared_sessions_or_replace_cosine_norms() {
    let index = index(25);
    let count_sidecars = || {
        let mut count = 0;
        index
            .visit_sidecars(|_, _| {
                count += 1;
                Ok(())
            })
            .unwrap();
        count
    };
    assert_eq!(count_sidecars(), 0);
    let metric = ScoreMetric::CosineSimilarity;
    // Base norms belong to explicit stage preparation, not query validation.
    let exact = index.exact_stage(metric).unwrap();
    let recipe = DynamicCascade {
        prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
        admission: &exact,
        rerank: &NoRerank,
    };
    assert_eq!(count_sidecars(), 1);
    assert!(
        index
            .search_batch(QueryInput::U8(&[1; 24]), controls(), recipe, None)
            .is_err()
    );
    assert!(
        index
            .search_batch(QueryInput::U8(&[0; 25]), controls(), recipe, None)
            .is_err()
    );
    assert!(
        index
            .search_batch(
                QueryInput::U8(&[1; 25]),
                DynamicSearchConfig { k: 0, ..controls() },
                recipe,
                None
            )
            .is_err()
    );
    for invalid in [f32::NAN, f32::INFINITY] {
        let mut query = vec![1.; 25];
        query[24] = invalid;
        assert!(
            index
                .search_batch(QueryInput::F32(&query), controls(), recipe, Some(metric))
                .is_err()
        );
    }
    assert_eq!(count_sidecars(), 1);
    let valid = index
        .search_batch(
            QueryInput::U8(&base(25)[..25]),
            controls(),
            recipe,
            Some(metric),
        )
        .unwrap();
    assert_eq!(valid.counts, vec![5]);
    assert_eq!(count_sidecars(), 1);
}

#[test]
fn canonical_exact_stages_support_both_transports_and_raw_cosine() {
    let byte_index = index(25);
    for metric in [
        ScoreMetric::SquaredL2,
        ScoreMetric::InnerProduct,
        ScoreMetric::CosineSimilarity,
    ] {
        let exact = byte_index.exact_stage(metric).unwrap();
        let recipe = DynamicCascade {
            prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
            admission: &exact,
            rerank: &NoRerank,
        };
        let data = base(25);
        let q = &data[..25];
        let floats: Vec<_> = q.iter().map(|&v| f32::from(v)).collect();
        let current = byte_index
            .search(QueryInput::U8(q), controls(), recipe, Some(metric))
            .unwrap();
        assert_eq!(
            current,
            byte_index
                .search_batch(QueryInput::F32(&floats), controls(), recipe, Some(metric))
                .unwrap()
        );
        assert_eq!(
            current,
            byte_index
                .search_batch(QueryInput::U8(q), controls(), recipe, Some(metric))
                .unwrap()
        );
    }
    // Exact f32 cosine is also well-defined at the low-level interface when
    // rows have different magnitudes; the highest dot product is not always
    // the most similar direction.
    let base = PaddedDataset::from_packed(
        &[100f32, 100., 2., 0., 0., 2.],
        PaddedLayout::for_kernel::<L2F32Distance>(2).unwrap(),
    )
    .unwrap();
    let graph = vec![
        (vec![1, 2], vec![], vec![]),
        (vec![0, 2], vec![], vec![]),
        (vec![0, 1], vec![], vec![]),
    ];
    let index = Orion::from_graph_partitions(base, &graph, 0, 2, 0).unwrap();
    let exact = index.exact_stage(ScoreMetric::CosineSimilarity).unwrap();
    let recipe = DynamicCascade {
        prefilter: &orion_ann_core::search::cascade::stage::NoPrefilter,
        admission: &exact,
        rerank: &NoRerank,
    };
    let result = index
        .search(
            QueryInput::F32(&[1., 0.]),
            DynamicSearchConfig {
                k: 1,
                search_list_size: 3,
                ..Default::default()
            },
            recipe,
            Some(ScoreMetric::CosineSimilarity),
        )
        .unwrap();
    assert_eq!(result.ids, vec![1]);
    assert_eq!(result.scores.unwrap().values, vec![1.]);
}
