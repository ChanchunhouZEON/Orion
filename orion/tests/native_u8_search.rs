//! Native-byte storage must reuse the production two-phase loop and exact scores.
use diskann::model::InmemDataset;
use orion::algorithm::search::diagnostics::{NeighborMode, SearchObserver};
use orion::algorithm::search::stage::admission::NativeU8Admission;
use orion::algorithm::search::stage::prefilter::NoPrefilter;
use orion::algorithm::search::stage::rerank::{F32Rerank, NoRerank};
use orion::algorithm::search::stage::{AdmissionSession, AdmissionStage};
use orion::model::Neighbor;
use orion::{CalibrationConfig, CalibrationMetric, Orion, PhasedGraph};

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

fn graph() -> PhasedGraph {
    let partitions: Vec<_> = (0..COUNT as u32)
        .map(|i| {
            (
                vec![(i + 1) % COUNT as u32, (i + 2) % COUNT as u32],
                vec![(i + 3) % COUNT as u32, (i + 4) % COUNT as u32],
                vec![(i + 17) % COUNT as u32, (i + 31) % COUNT as u32],
            )
        })
        .collect();
    PhasedGraph::build_from_partitions(&partitions, 4, 2)
}

// Independent full-precision reference admission. Deliberately no quantizer.
struct ExactFloat<'a>(&'a InmemDataset<f32, DIM>);
struct FloatSession<'a> {
    dataset: &'a InmemDataset<f32, DIM>,
    query: [f32; DIM],
}
impl AdmissionStage<DIM> for ExactFloat<'_> {
    fn open<'a>(&'a self, query: &[f32; DIM]) -> Box<dyn AdmissionSession + 'a> {
        Box::new(FloatSession {
            dataset: self.0,
            query: *query,
        })
    }
}
impl AdmissionSession for FloatSession<'_> {
    fn entry_distance(&self, id: u32) -> f32 {
        let row = &self.dataset.data[id as usize * DIM..(id as usize + 1) * DIM];
        row.iter()
            .zip(self.query)
            .map(|(&a, b)| (a - b) * (a - b))
            .sum()
    }
    unsafe fn admit_stream(&self, ids: &[u32], out: *mut Neighbor, cutoff: f32, _: usize) -> usize {
        let mut count = 0;
        for &id in ids {
            let distance = self.entry_distance(id);
            if distance < cutoff {
                unsafe {
                    out.add(count).write(Neighbor::new(id, distance));
                }
                count += 1;
            }
        }
        count
    }
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
    let query = std::array::from_fn(|j| floats.data[7 * DIM + j]);
    let admission = NativeU8Admission::new(&bytes);
    let session = admission.open(&query);
    let oracle = FloatSession {
        dataset: &floats,
        query,
    };
    for count in [0, 1, 3, 16, COUNT] {
        let ids: Vec<_> = (0..count as u32).rev().collect();
        for cutoff in [0.0, 10_000.0, f32::MAX] {
            let mut output = vec![Neighbor::new(0, 0.0); count];
            let written = unsafe { session.admit_stream(&ids, output.as_mut_ptr(), cutoff, 4) };
            let expected: Vec<_> = ids
                .iter()
                .filter_map(|&id| {
                    let distance = oracle.entry_distance(id);
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
    let (bytes, floats) = datasets();
    let pointer = bytes.data.as_ptr();
    let native = Orion::from_phased_graph(bytes, graph(), 0, None, None);
    let reference = Orion::from_phased_graph(floats, graph(), 0, None, None);
    assert_eq!(native.dataset.data.as_ptr(), pointer);
    assert_eq!(native.dataset.data.len(), COUNT * DIM + 64);
    let queries: Vec<[f32; DIM]> = [7, 13, 41]
        .iter()
        .map(|&i| std::array::from_fn(|j| reference.dataset.data[i * DIM + j]))
        .collect();
    let target = CalibrationConfig {
        k: 5,
        metric: CalibrationMetric::L2,
    };
    let actual = native.calibrate(&queries, 32, 3, target).unwrap();
    let expected = reference.calibrate(&queries, 32, 3, target).unwrap();
    assert_eq!(actual.threshold, expected.threshold);
    assert_eq!(actual.early_exit_limit, expected.early_exit_limit);
    let mut had_post_expansion = false;
    for query in &queries {
        let mut native_trace = Trace::default();
        let mut float_trace = Trace::default();
        let result = native
            .search_unified_observed(
                query,
                5,
                32,
                1,
                1e6,
                usize::MAX,
                None::<&NoPrefilter>,
                &NativeU8Admission::new(&native.dataset),
                &NoRerank,
                NeighborMode::LocalExtra,
                &mut native_trace,
            )
            .unwrap();
        let expected = reference
            .search_unified_observed(
                query,
                5,
                32,
                1,
                1e6,
                usize::MAX,
                None::<&NoPrefilter>,
                &ExactFloat(&reference.dataset),
                &F32Rerank::new(&reference.dataset),
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
    NativeU8Admission::new(&bytes).open(&[0.5; DIM]);
}
