/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::*;
use crate::{
    CompressedIndex, Orion,
    model::Neighbor,
    search::cascade::stage::{
        NoPrefilter,
        admission::L2U8Admission,
        rerank::{DynamicF32Rerank, NoRerank},
    },
};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_platform::region::{ArrayView, ReadRegion};
use orion_ann_vector::{L2F32Distance, PaddedLayout};
use std::{
    future::Future,
    io::BufWriter,
    pin::pin,
    task::{Context, Poll, Waker},
};
fn ready<T>(f: impl Future<Output = T>) -> T {
    let mut f = pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("resident read unexpectedly suspended"),
    }
}
fn index(points: usize, dim: usize) -> Orion {
    let packed: Vec<_> = (0..points * dim)
        .map(|i| ((i * 31 + i / dim * 7) % 257) as f32 / 17.)
        .collect();
    let base = PaddedDataset::from_packed(
        &packed,
        PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap(),
    )
    .unwrap();
    let parts: Vec<_> = (0..points as u32)
        .map(|i| {
            (
                (1..5).map(|j| (i + j * 7) % points as u32).collect(),
                (5..8).map(|j| (i + j * 7) % points as u32).collect(),
                vec![(i + 1) % points as u32],
            )
        })
        .collect();
    Orion::from_graph_partitions(base, &parts, 0, 8, 1).unwrap()
}
fn graph<D: crate::index::IndexData>(index: &CompressedIndex<D>) -> ReadRegion {
    let mut output = BufWriter::new(Vec::new());
    index.encode_native_graph(Vec::new(), &mut output).unwrap();
    ArrayView::copy_from(&output.into_inner().unwrap(), &Default::default())
        .unwrap()
        .region()
        .clone()
}
#[derive(Default, Debug, PartialEq)]
struct Trace {
    hops: Vec<(u32, bool, bool)>,
    candidates: Vec<Vec<u32>>,
    beams: Vec<Vec<(u32, u32)>>,
    admits: Vec<(usize, Vec<(u32, u32)>)>,
    exits: usize,
}
impl SearchObserver for Trace {
    fn expansion_started(&mut self, id: u32, converged: bool, peel: bool) {
        self.hops.push((id, converged, peel));
    }
    fn expansion(&mut self, _: bool, ids: &[u32]) {
        self.candidates.push(ids.to_vec());
    }
    fn beam_updated(&mut self, beam: &[Neighbor]) {
        self.beams
            .push(beam.iter().map(|n| (n.id, n.distance.to_bits())).collect());
    }
    fn admission(&mut self, n: usize, beam: &[Neighbor]) {
        self.admits.push((
            n,
            beam.iter().map(|n| (n.id, n.distance.to_bits())).collect(),
        ));
    }
    fn early_exit(&mut self) {
        self.exits += 1;
    }
}
fn reference<D: CascadeData>(
    index: &CompressedIndex<D>,
    raw: QueryInput<'_>,
    config: DynamicSearchConfig,
    recipe: DynamicCascade<'_>,
    trace: &mut impl SearchObserver,
) -> (Vec<u32>, Option<Vec<f32>>) {
    let plan = CascadePlan::new(index.dataset(), recipe, RowTransform::Identity);
    plan.validate().unwrap();
    let q = plan.prepare(raw).unwrap();
    let filter = plan.prefilter(&q);
    let admission = plan.admission(&q);
    index
        .search_sessions_reference(
            config.k,
            config.search_list_size,
            config.window_size,
            config.epsilon,
            config.early_exit_limit,
            filter.as_ref(),
            admission.session(),
            NeighborMode::default(),
            trace,
            SetupTimer::now(),
            |scratch, factor| {
                plan.finish(&q, scratch, config.k, factor, Some(ScoreMetric::SquaredL2))
            },
        )
        .unwrap()
}
#[test]
fn original_resident_and_suspendable_loops_preserve_order_and_scores() {
    for dim in [25, 128, 960] {
        let index = index(128, dim);
        let driver = ready(SearchDriver::open(graph(&index), 0)).unwrap();
        let exact = index.exact_stage(ScoreMetric::SquaredL2).unwrap();
        let quant = L2U8Admission::new(index.ensure_quantized_dataset());
        let filter = crate::search::cascade::stage::prefilter::jl::JlPrefilter::new(
            index.ensure_quantized_dataset_jl(),
        );
        let rerank = DynamicF32Rerank::<L2F32Distance>::new(index.dataset());
        for recipe in [
            DynamicCascade {
                prefilter: &NoPrefilter,
                admission: &exact,
                rerank: &NoRerank,
            },
            DynamicCascade {
                prefilter: &NoPrefilter,
                admission: &quant,
                rerank: &rerank,
            },
            DynamicCascade {
                prefilter: &filter,
                admission: &quant,
                rerank: &rerank,
            },
        ] {
            for l in [8, 24, 64] {
                for epsilon in [0., 0.5, 1.] {
                    for row in [0, 17, 47] {
                        let q: Vec<_> = (0..dim)
                            .map(|j| ((row * dim + j) * 31 + row * 7) % 257)
                            .map(|v| v as f32 / 17. + 0.01)
                            .collect();
                        let cfg = DynamicSearchConfig {
                            k: 5,
                            search_list_size: l,
                            epsilon,
                            early_exit_limit: 4,
                            ..Default::default()
                        };
                        let mut old = Trace::default();
                        let expected =
                            reference(&index, QueryInput::F32(&q), cfg, recipe, &mut old);
                        let mut now = Trace::default();
                        let ids = index
                            .search_observed(
                                QueryInput::F32(&q),
                                cfg,
                                recipe,
                                NeighborMode::default(),
                                &mut now,
                            )
                            .unwrap();
                        assert_eq!(expected.0, ids);
                        assert_eq!(old, now, "resident scheduling changed");
                        let mut disk = Trace::default();
                        let actual = ready(driver.search_observed(
                            index.dataset(),
                            QueryInput::F32(&q),
                            cfg,
                            recipe,
                            RowTransform::Identity,
                            Some(ScoreMetric::SquaredL2),
                            &mut disk,
                        ))
                        .unwrap();
                        assert_eq!(actual.ids, expected.0);
                        assert_eq!(actual.scores.unwrap().values, expected.1.unwrap());
                        assert_eq!(old, disk, "storage scheduling changed");
                    }
                }
            }
        }
    }
}
#[test]
fn malformed_headers_and_records_fail_before_computation() {
    let index = index(128, 32);
    let raw = graph(&index).as_bytes().to_vec();
    for (offset, value) in [(0, 0u32), (8, 1), (16, 500), (28, 1), (32, u32::MAX)] {
        let mut bytes = raw.clone();
        bytes[offset..offset + 4].copy_from_slice(&value.to_ne_bytes());
        let region = ArrayView::copy_from(&bytes, &Default::default())
            .unwrap()
            .region()
            .clone();
        match ready(SearchDriver::open(region, 0)) {
            Err(_) => assert!(offset < 16),
            Ok(driver) => {
                let exact = index.exact_stage(ScoreMetric::SquaredL2).unwrap();
                assert!(
                    ready(driver.search(
                        index.dataset(),
                        QueryInput::F32(&[1.; 32]),
                        DynamicSearchConfig::default(),
                        DynamicCascade {
                            prefilter: &NoPrefilter,
                            admission: &exact,
                            rerank: &NoRerank
                        },
                        RowTransform::Identity,
                        None
                    ))
                    .is_err()
                );
            }
        }
    }
}
// Ignored in normal CI: paired original/shared traversal timing, no disk effects.
#[test]
#[ignore = "release-only traversal performance comparison"]
fn resident_traversal_performance() {
    use std::{
        hint::black_box,
        time::{Duration, Instant},
    };
    for dim in [128, 960] {
        let index = index(8192, dim);
        let exact = index.exact_stage(ScoreMetric::SquaredL2).unwrap();
        let recipe = DynamicCascade {
            prefilter: &NoPrefilter,
            admission: &exact,
            rerank: &NoRerank,
        };
        let cfg = DynamicSearchConfig {
            k: 10,
            search_list_size: 64,
            ..Default::default()
        };
        let queries: Vec<Vec<f32>> = (0..64)
            .map(|r| {
                (0..dim)
                    .map(|j| ((r * dim + j) * 31 + r * 7) % 257)
                    .map(|v| v as f32 / 17. + 0.01)
                    .collect()
            })
            .collect();
        for pair in 0..3 {
            let mut times = [0.; 2];
            for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
                let start = Instant::now();
                let mut n = 0;
                while start.elapsed() < Duration::from_secs(1) {
                    for q in &queries {
                        let plan =
                            CascadePlan::new(index.dataset(), recipe, RowTransform::Identity);
                        let prepared = plan.prepare(QueryInput::F32(q)).unwrap();
                        let f = plan.prefilter(&prepared);
                        let a = plan.admission(&prepared);
                        let result = if arm == 0 {
                            index.search_sessions_reference(
                                cfg.k,
                                cfg.search_list_size,
                                cfg.window_size,
                                cfg.epsilon,
                                cfg.early_exit_limit,
                                f.as_ref(),
                                a.session(),
                                NeighborMode::default(),
                                &mut NoopObserver,
                                SetupTimer::now(),
                                |s, factor| plan.finish(&prepared, s, cfg.k, factor, None),
                            )
                        } else {
                            index.search_sessions(
                                cfg.k,
                                cfg.search_list_size,
                                cfg.window_size,
                                cfg.epsilon,
                                cfg.early_exit_limit,
                                f.as_ref(),
                                a.session(),
                                NeighborMode::default(),
                                &mut NoopObserver,
                                SetupTimer::now(),
                                |s, factor| plan.finish(&prepared, s, cfg.k, factor, None),
                            )
                        };
                        black_box(result.unwrap());
                        n += 1;
                    }
                }
                times[arm] = start.elapsed().as_secs_f64() / n as f64;
            }
            println!(
                "dimension={dim} pair={pair} original_us={} shared_us={} change_pct={}",
                times[0] * 1e6,
                times[1] * 1e6,
                (times[1] / times[0] - 1.) * 100.
            );
        }
    }
}
