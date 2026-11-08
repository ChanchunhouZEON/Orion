/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Native-byte storage must reuse the production two-phase loop and exact scores.
#[path = "support/index_graph.rs"]
mod graph_fixture;
use graph_fixture::{GraphFixture, from_fixture_graph};
use orion_ann_core::algorithm::search::diagnostics::{NeighborMode, SearchObserver};
use orion_ann_core::algorithm::search::stage::Admission;
use orion_ann_core::algorithm::search::stage::admission::NativeU8Admission;
use orion_ann_core::algorithm::search::stage::rerank::F32Rerank;
use orion_ann_core::model::Neighbor;
use orion_ann_core::{
    CalibrationConfig, CalibrationMetric, DynamicCascade, DynamicSearchConfig, PaddedDataset,
    PaddedLayout,
};
use orion_ann_diskann::model::InmemDataset;

const DIM: usize = 128;
const COUNT: usize = 64;

fn datasets() -> (InmemDataset<u8, DIM>, InmemDataset<f32, DIM>) {
    let mut bytes = InmemDataset::<u8, DIM>::new(COUNT, 1.0).unwrap();
    let mut floats = InmemDataset::<f32, DIM>::new(COUNT, 1.0).unwrap();
    for i in 0..COUNT * DIM {
        // A restricted range catches accidental affine rescaling to [0, 255].
        let value = 80 + ((i / DIM * 17 + i % DIM * 7) % 101) as u8;
        bytes.data[i] = value;
        floats.data[i] = f32::from(value);
    }
    (bytes, floats)
}

fn graph() -> GraphFixture {
    let partitions: Vec<_> = (0..COUNT as u32)
        .map(|i| {
            (
                vec![(i + 1) % COUNT as u32, (i + 2) % COUNT as u32],
                vec![(i + 3) % COUNT as u32, (i + 4) % COUNT as u32],
                vec![(i + 17) % COUNT as u32, (i + 31) % COUNT as u32],
            )
        })
        .collect();
    GraphFixture::build_from_partitions(&partitions, 4, 2)
}

#[derive(Default, Debug, PartialEq)]
struct Trace(Vec<(bool, Vec<u32>)>);
impl SearchObserver for Trace {
    fn expansion(&mut self, converged: bool, ids: &[u32]) {
        self.0.push((converged, ids.to_vec()));
    }
}

#[test]
fn native_admission_scores_and_cutoffs_equal_full_precision_without_rescaling() {
    let (bytes, floats) = datasets();
    let query: [f32; DIM] = std::array::from_fn(|j| floats.data[7 * DIM + j]);
    let admission = NativeU8Admission::new(&bytes);
    let session = (&admission as &Admission).open(&query);
    for count in [0, 1, 3, 16, COUNT] {
        let ids: Vec<_> = (0..count as u32).rev().collect();
        for cutoff in [0.0, 10_000.0, f32::MAX] {
            let mut output = vec![Neighbor::new(0, 0.0); count];
            let written = unsafe { session.admit_stream(&ids, output.as_mut_ptr(), cutoff, 4) };
            let expected: Vec<_> = ids
                .iter()
                .filter_map(|&id| {
                    let row = &floats.data[id as usize * DIM..][..DIM];
                    let distance = row
                        .iter()
                        .zip(query)
                        .map(|(x, y)| (x - y) * (x - y))
                        .sum::<f32>();
                    assert_eq!(session.entry_distance(id), distance);
                    (distance < cutoff).then_some((id, distance))
                })
                .collect();
            assert_eq!(
                output[..written]
                    .iter()
                    .map(|n| (n.id, n.distance))
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }
}

#[test]
fn native_storage_preserves_calibration_search_and_local_extra_expansion() {
    let (reference_bytes, floats) = datasets();
    let bytes = PaddedDataset::from_packed(
        &reference_bytes.data[..COUNT * DIM],
        PaddedLayout::for_kernel::<orion_ann_vector::L2U8Distance>(DIM).unwrap(),
    )
    .unwrap();
    let pointer = bytes.as_slice().as_ptr();
    let native = from_fixture_graph(bytes, graph(), 0);
    let reference = from_fixture_graph(floats, graph(), 0);
    assert_eq!(native.dataset().as_slice().as_ptr(), pointer);
    assert_eq!(native.dataset().len(), COUNT);
    assert_eq!(native.dataset().layout().logical_dim(), DIM);
    let queries: Vec<[f32; DIM]> = [7, 13, 41]
        .iter()
        .map(|&i| std::array::from_fn(|j| reference.dataset().data[i * DIM + j]))
        .collect();
    let target = CalibrationConfig {
        k: 5,
        metric: CalibrationMetric::L2,
    };
    let actual = native
        .calibrate(queries.as_flattened(), 32, 3, target)
        .unwrap();
    let expected = reference.calibrate(&queries, 32, 3, target).unwrap();
    assert_eq!(actual.threshold, expected.threshold);
    assert_eq!(actual.early_exit_limit, expected.early_exit_limit);
    // A window of one and epsilon=1 trigger the post-convergence path while
    // remaining within the current API's validated convergence range.
    let controls = DynamicSearchConfig {
        k: 5,
        search_list_size: 32,
        window_size: 1,
        epsilon: 1.0,
        early_exit_limit: usize::MAX,
    };
    assert_eq!(NeighborMode::default(), NeighborMode::LocalExtra);
    let mut had_post_expansion = false;
    for query in &queries {
        let mut native_trace = Trace::default();
        let mut float_trace = Trace::default();
        let byte_query: Vec<u8> = query.iter().map(|&value| value as u8).collect();
        let result = native
            .search_l2_observed(&byte_query, controls, &mut native_trace)
            .unwrap();
        let expected = reference
            .search_cascade_observed(
                query,
                controls,
                DynamicCascade {
                    prefilter: None,
                    // Byte scores are exact; the separate scalar test checks their values.
                    admission: &NativeU8Admission::new(&reference_bytes),
                    rerank: &F32Rerank::new(reference.dataset()),
                },
                NeighborMode::LocalExtra,
                &mut float_trace,
            )
            .unwrap();
        assert_eq!(native_trace, float_trace);
        assert_eq!(result, expected);
        had_post_expansion |= native_trace
            .0
            .iter()
            .any(|(converged, ids)| *converged && !ids.is_empty());
    }
    assert!(
        had_post_expansion,
        "fixture must exercise post-convergence candidates"
    );
}

#[test]
#[should_panic(expected = "original byte-valued queries")]
fn native_admission_rejects_fractional_queries() {
    let (bytes, _) = datasets();
    (&NativeU8Admission::new(&bytes) as &Admission).open(&[0.5; DIM]);
}
