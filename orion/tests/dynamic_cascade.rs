//! Compare runtime-dimensional recipes with the compatibility facade, including
//! traversal work rather than only top-k IDs. Positive-only rows verify that
//! zero padding does not leak into statistical fitting.
use diskann::model::{InmemDataset, PaddedDataset};
use orion::algorithm::search::{
    calibrate::{CalibrationDiagnostics, CalibrationMetric},
    diagnostics::{NeighborMode, SearchTrace},
    stage::{
        AdmissionStage, PrefilterStage, RerankStage,
        admission::*,
        prefilter::{
            jl::{JlMipsPrefilter, JlPrefilter},
            jl_hadamard::JlHadamardPrefilter,
            rabitq::RabitqPrefilter,
        },
        rerank::{DynamicF32Rerank, F32Rerank, IpF32Rerank, NoRerank, U16Rerank},
    },
};
use orion::model::dataset::*;
use orion::{
    CalibrationConfig, DynamicCascade, Orion, DynamicSearchConfig, StaticOrion, PhasedGraph,
};
use vector::{IpF32Distance, L2F32Distance, L2U8Distance, PaddedLayout};
const ROWS: usize = 48;
fn values(dim: usize) -> Vec<f32> {
    (0..ROWS * dim)
        .map(|i| ((i * 31 + i / dim * 7) % 103) as f32 / 17. - 3.)
        .collect()
}
fn graph() -> PhasedGraph {
    let parts: Vec<_> = (0..ROWS as u32)
        .map(|i| {
            (
                (1..5).map(|j| (i + j * 7) % ROWS as u32).collect(),
                (5..8).map(|j| (i + j * 7) % ROWS as u32).collect(),
                vec![(i + 1) % ROWS as u32],
            )
        })
        .collect();
    PhasedGraph::build_from_partitions(&parts, 8, 1)
}
fn dynamic(dim: usize) -> Orion {
    Orion::from_phased_graph(
        PaddedDataset::from_packed(
            &values(dim),
            PaddedLayout::for_kernel::<L2F32Distance>(dim).unwrap(),
        )
        .unwrap(),
        graph(),
        0,
        None,
        None,
    )
}
fn typed<const N: usize>() -> StaticOrion<N, f32> {
    let mut base = InmemDataset::new(ROWS, 1.).unwrap();
    base.data[..ROWS * N].copy_from_slice(&values(N));
    StaticOrion::from_phased_graph(base, graph(), 0, None, None)
}
fn compare<const N: usize>(
    old: &StaticOrion<N, f32>,
    new: &Orion,
    a: &dyn AdmissionStage,
    b: &dyn AdmissionStage,
    pa: Option<&dyn PrefilterStage>,
    pb: Option<&dyn PrefilterStage>,
    ra: &dyn RerankStage,
    rb: &dyn RerankStage,
) {
    for l in [8, 24] {
        for epsilon in [0., 0.5, 1.] {
            for mode in NeighborMode::ALL {
                for row in [0, 17, 47] {
                    let q: [f32; N] = std::array::from_fn(|j| old.dataset.data[row * N + j] + 0.01);
                    let cfg = DynamicSearchConfig {
                        k: 5,
                        search_list_size: l,
                        epsilon,
                        early_exit_limit: 4,
                        ..Default::default()
                    };
                    let mut x = SearchTrace::new(&[0, 17, 47]);
                    let mut y = SearchTrace::new(&[0, 17, 47]);
                    let expected = old
                        .search_unified_observed(
                            &q,
                            5,
                            l,
                            cfg.window_size,
                            epsilon,
                            4,
                            pa,
                            a,
                            ra,
                            mode,
                            &mut x,
                        )
                        .unwrap();
                    let result = new
                        .search_cascade_observed(
                            &q,
                            cfg,
                            DynamicCascade {
                                prefilter: pb,
                                admission: b,
                                rerank: rb,
                            },
                            mode,
                            &mut y,
                        )
                        .unwrap();
                    assert_eq!(expected, result, "L={l} eps={epsilon} mode={mode:?}");
                    assert_eq!(format!("{x:?}"), format!("{y:?}"), "search work changed");
                }
            }
        }
    }
}
#[test]
fn typed_and_dynamic_cascades_preserve_ids_and_traversal() {
    let old = typed::<32>();
    let new = dynamic(32);
    let oa = L2U8Admission::new(old.ensure_quantized_dataset());
    let na = L2U8Admission::new(new.ensure_quantized_dataset());
    let of = F32Rerank::new(&old.dataset);
    let nf = DynamicF32Rerank::<L2F32Distance>::new(&new.dataset);
    compare(&old, &new, &oa, &na, None, None, &of, &nf);
    compare(
        &old,
        &new,
        &L2U16Admission::new(old.ensure_quantized_dataset_l2_u16()),
        &L2U16Admission::new(new.ensure_quantized_dataset_l2_u16()),
        None,
        None,
        &U16Rerank::new(old.ensure_quantized_dataset_l2_u16()),
        &U16Rerank::new(new.ensure_quantized_dataset_l2_u16()),
    );
    compare(
        &old,
        &new,
        &L2KTAdmission::new(old.ensure_quantized_dataset_l2_kt()),
        &L2KTAdmission::new(new.ensure_quantized_dataset_l2_kt()),
        None,
        None,
        &of,
        &nf,
    );
    compare(
        &old,
        &new,
        &oa,
        &na,
        Some(&JlPrefilter::new(old.ensure_quantized_dataset_jl())),
        Some(&JlPrefilter::new(new.ensure_quantized_dataset_jl())),
        &of,
        &nf,
    );
    compare(
        &old,
        &new,
        &oa,
        &na,
        Some(&JlHadamardPrefilter::new(
            old.ensure_quantized_dataset_jl_hadamard(),
        )),
        Some(&JlHadamardPrefilter::new(
            new.ensure_quantized_dataset_jl_hadamard(),
        )),
        &of,
        &nf,
    );
    compare(
        &old,
        &new,
        &oa,
        &na,
        Some(&RabitqPrefilter::new(old.ensure_quantized_dataset_rabitq())),
        Some(&RabitqPrefilter::new(new.ensure_quantized_dataset_rabitq())),
        &of,
        &nf,
    );
    let oi = IpF32Rerank::new(&old.dataset);
    let ni = DynamicF32Rerank::<IpF32Distance>::new(&new.dataset);
    compare(
        &old,
        &new,
        &MipsI8Admission::new(old.ensure_quantized_dataset_mips()),
        &MipsI8Admission::new(new.ensure_quantized_dataset_mips()),
        Some(&JlMipsPrefilter::new(
            old.ensure_quantized_dataset_jl_mips(),
        )),
        Some(&JlMipsPrefilter::new(
            new.ensure_quantized_dataset_jl_mips(),
        )),
        &oi,
        &ni,
    );
    compare(
        &old,
        &new,
        &MipsI16Admission::new(old.ensure_quantized_dataset_mips_i16()),
        &MipsI16Admission::new(new.ensure_quantized_dataset_mips_i16()),
        None,
        None,
        &oi,
        &ni,
    );
}
fn same_calibration(a: &CalibrationDiagnostics, b: &CalibrationDiagnostics) {
    assert_eq!(a.params.threshold, b.params.threshold);
    assert_eq!(a.params.early_exit_limit, b.params.early_exit_limit);
    assert_eq!(a.admission_rates, b.admission_rates);
    assert_eq!(a.topk_coverage_by_step, b.topk_coverage_by_step);
    assert_eq!(a.useful_gaps, b.useful_gaps);
    assert_eq!(a.tail_gaps, b.tail_gaps);
}
fn calibration<const N: usize>() {
    let old = typed::<N>();
    let new = dynamic(N);
    let queries: Vec<[f32; N]> = (0..6)
        .map(|i| std::array::from_fn(|j| old.dataset.data[i * N + j] + 0.1))
        .collect();
    let packed: Vec<_> = queries.iter().flatten().copied().collect();
    for metric in [
        CalibrationMetric::L2,
        CalibrationMetric::InnerProduct,
        CalibrationMetric::Cosine,
    ] {
        let cfg = CalibrationConfig { k: 5, metric };
        let a = old
            .calibrate_with_diagnostics(&queries, 16, 3, cfg)
            .unwrap();
        let b = new.calibrate_with_diagnostics(&packed, 16, 3, cfg).unwrap();
        same_calibration(&a, &b);
        let a = old.calibrate(&queries, 16, 3, cfg).unwrap();
        let b = new.calibrate(&packed, 16, 3, cfg).unwrap();
        assert_eq!(a.threshold, b.threshold);
        assert_eq!(a.early_exit_limit, b.early_exit_limit);
    }
}
#[test]
fn dynamic_calibration_matches_typed_metrics_and_odd_dimensions() {
    calibration::<3>();
    calibration::<32>();
    calibration::<129>();
}
#[test]
fn native_u8_calibration_and_errors() {
    let packed: Vec<u8> = values(33).iter().map(|v| ((v + 3.) * 17.) as u8).collect();
    let mut base = InmemDataset::<u8, 33>::new(ROWS, 1.).unwrap();
    base.data[..packed.len()].copy_from_slice(&packed);
    let old = StaticOrion::from_phased_graph(base, graph(), 0, None, None);
    let new = Orion::from_phased_graph(
        PaddedDataset::from_packed(
            &packed,
            PaddedLayout::for_kernel::<L2U8Distance>(33).unwrap(),
        )
        .unwrap(),
        graph(),
        0,
        None,
        None,
    );
    let qs: Vec<[f32; 33]> = packed
        .chunks_exact(33)
        .take(5)
        .map(|q| std::array::from_fn(|j| f32::from(q[j])))
        .collect();
    let flat: Vec<_> = qs.iter().flatten().copied().collect();
    let cfg = CalibrationConfig {
        k: 5,
        ..Default::default()
    };
    same_calibration(
        &old.calibrate_with_diagnostics(&qs, 16, 3, cfg).unwrap(),
        &new.calibrate_with_diagnostics(&flat, 16, 3, cfg).unwrap(),
    );
    assert!(
        new.calibrate(
            &flat,
            16,
            3,
            CalibrationConfig {
                metric: CalibrationMetric::Cosine,
                ..cfg
            }
        )
        .is_err()
    );
    assert!(new.calibrate(&flat[..32], 16, 3, cfg).is_err());
    assert!(new.calibrate(&[f32::NAN; 33], 16, 3, cfg).is_err());
    assert!(new.calibrate(&flat, 16, 0, cfg).is_err());
    assert!(new.calibrate(&flat, 4, 3, cfg).is_err());
}
#[test]
fn odd_dimensions_cascade_batch_cache_and_padding() {
    let dir =
        std::env::temp_dir().join(format!("dynamic-sidecar-roundtrip-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for dim in [1, 3, 31, 33, 129] {
        let index = dynamic(dim);
        macro_rules! roundtrip {
            ($name:literal,$build:expr,$ty:ty) => {{
                let ds = $build;
                let path = dir.join($name);
                let second = dir.join(concat!($name, ".reload"));
                ds.save(&path).unwrap();
                let loaded = <$ty>::load(&path).unwrap();
                loaded.save(&second).unwrap();
                assert_eq!(std::fs::read(path).unwrap(), std::fs::read(second).unwrap());
                loaded
            }};
        }
        let u8s = roundtrip!(
            "u8",
            QuantizedDataset::<L2U8>::from_f32_dataset(&index.dataset),
            QuantizedDataset<L2U8>
        );
        roundtrip!(
            "u16",
            QuantizedDataset::<L2U16>::from_f32_dataset(&index.dataset),
            QuantizedDataset<L2U16>
        );
        roundtrip!(
            "i8",
            QuantizedDataset::<MipsI8>::from_f32_dataset(&index.dataset),
            QuantizedDataset<MipsI8>
        );
        roundtrip!(
            "i16",
            QuantizedDataset::<MipsI16>::from_f32_dataset(&index.dataset),
            QuantizedDataset<MipsI16>
        );
        roundtrip!("kt", L2KTDataset::build_from(&index.dataset), L2KTDataset);
        let jl = roundtrip!(
            "jl",
            JLSparseDataset::<1024>::build_from(&index.dataset, 42),
            JLSparseDataset<1024>
        );
        roundtrip!(
            "jl-mips",
            JLSparseDatasetMips::<1024>::build_from(&index.dataset, 42),
            JLSparseDatasetMips<1024>
        );
        roundtrip!(
            "hadamard",
            JlHadamardDataset::build_from(&index.dataset, 42),
            JlHadamardDataset
        );
        roundtrip!(
            "rb",
            rabitq_dataset::RabitQDataset::build_from(&index.dataset, 42),
            rabitq_dataset::RabitQDataset
        );
        roundtrip!(
            "rb4",
            RabitQ4Dataset::build_from(&index.dataset, 42),
            RabitQ4Dataset
        );
        let admission = L2U8Admission::new(&u8s);
        let filter = JlPrefilter::new(&jl);
        let rerank = DynamicF32Rerank::<L2F32Distance>::new(&index.dataset);
        let recipe = DynamicCascade {
            prefilter: Some(&filter),
            admission: &admission,
            rerank: &rerank,
        };
        let cfg = DynamicSearchConfig {
            k: 5,
            search_list_size: 24,
            ..Default::default()
        };
        let qs = &values(dim)[..dim * 4];
        let batch = index.search_batch_cascade(qs, cfg, recipe).unwrap();
        for (q, ids) in qs.chunks_exact(dim).zip(batch) {
            assert_eq!(index.search_cascade(q, cfg, recipe).unwrap(), ids);
        }
        assert!(
            index
                .search_batch_cascade(&[], cfg, recipe)
                .unwrap()
                .is_empty()
        );
        assert!(
            index
                .search_cascade(&vec![0.; dim + 1], cfg, recipe)
                .is_err()
        );
        // A foreign sidecar is rejected before opening a session or streaming.
        let foreign = dynamic(dim + 1);
        let bad = L2U8Admission::new(foreign.ensure_quantized_dataset());
        assert!(
            index
                .search_cascade(
                    &qs[..dim],
                    cfg,
                    DynamicCascade {
                        admission: &bad,
                        rerank: &NoRerank,
                        prefilter: None
                    }
                )
                .is_err()
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}
#[test]
fn padding_does_not_affect_quantization_or_projection_width() {
    let packed = vec![5.0f32, 6., 7., 8., 9., 10.];
    let base =
        PaddedDataset::from_packed(&packed, PaddedLayout::new::<f32>(3, 256).unwrap()).unwrap();
    let q = QuantizedDataset::<L2U8>::from_f32_dataset(&base);
    let kt = L2KTDataset::build_from(&base);
    assert_eq!(q.params.slope, kt.params.slope);
    assert_eq!(q.params.offset, kt.params.offset);
    let expected = QuantParamsL2::from_range(5., 10.);
    assert_eq!(q.params.slope, expected.slope);
    assert_eq!(q.params.offset, expected.offset);
    for id in 0..2 {
        assert_eq!(
            kt.norms_sq[id],
            kt.data.as_slice()[id * kt.stride..id * kt.stride + 3]
                .iter()
                .map(|&x| i32::from(x).pow(2))
                .sum::<i32>()
        );
    }
    let large = dynamic(1025);
    assert_eq!(large.ensure_quantized_dataset_jl_hadamard().d_pad, 2048);
}

#[test]
fn odd_quantized_entry_and_stream_distances_match_scalar() {
    use orion::model::Neighbor;
    for dim in [1, 3, 31, 33, 65, 129] {
        let index = dynamic(dim);
        let q = &values(dim)[dim..2 * dim];
        let ids = [47, 0, 23, 46];
        macro_rules! check {
            ($spec:ty,$admission:ident,$score:expr) => {{
                let ds = QuantizedDataset::<$spec>::from_f32_dataset(&index.dataset);
                let query = ds.quantize_query(q);
                let stage = $admission::new(&ds);
                let session = stage.open(q);
                let expected: Vec<f32> = ids
                    .iter()
                    .map(|&id| {
                        let row = unsafe { ds.get_vertex_unchecked(id) };
                        row.iter()
                            .zip(&query)
                            .map(|(&a, &b)| ($score)(a as f64, b as f64))
                            .sum::<f64>() as f32
                    })
                    .collect();
                let mut out = vec![Neighbor::new(0, 0.); ids.len()];
                let count =
                    unsafe { session.admit_stream(&ids, out.as_mut_ptr(), f32::INFINITY, 32) };
                assert_eq!(count, ids.len());
                for ((&id, &d), got) in ids.iter().zip(&expected).zip(&out) {
                    assert_eq!(got.id, id);
                    // Large u16/i16 sums round to f32 in the kernel reduction.
                    let tolerance = d.abs().max(1.) * 2e-6;
                    assert!(
                        (got.distance - d).abs() <= tolerance,
                        "dim={dim} expected={d} got={}",
                        got.distance
                    );
                    assert!((session.entry_distance(id) - d).abs() <= tolerance);
                }
            }};
        }
        check!(L2U8, L2U8Admission, |a: f64, b: f64| (a - b) * (a - b));
        check!(L2U16, L2U16Admission, |a: f64, b: f64| (a - b) * (a - b));
        check!(MipsI8, MipsI8Admission, |a: f64, b: f64| -a * b);
        check!(MipsI16, MipsI16Admission, |a: f64, b: f64| -a * b);
        let ds = L2KTDataset::build_from(&index.dataset);
        let query = ds.quantize_query(q);
        let stage = L2KTAdmission::new(&ds);
        let session = stage.open(q);
        for id in ids {
            let expected: i32 = ds.data.as_slice()
                [id as usize * ds.stride..id as usize * ds.stride + dim]
                .iter()
                .zip(&query)
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).pow(2))
                .sum();
            assert_eq!(session.entry_distance(id), expected as f32);
        }
    }
}
#[test]
fn sidecar_cache_rebuilds_wrong_shapes_and_rejects_bad_headers() {
    let dir = std::env::temp_dir().join(format!("dynamic-sidecar-invalid-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("index.qds");
    dynamic(32).ensure_quantized_dataset().save(&path).unwrap();
    let mut index = dynamic(33);
    index.cache_base_path = dir.join("index.bin");
    assert_eq!(index.ensure_quantized_dataset().dimension, 33);
    assert_eq!(QuantizedDataset::<L2U8>::load(&path).unwrap().dimension, 33);
    let original = std::fs::read(&path).unwrap();
    for (offset, value) in [(4, 99u32), (8, u32::MAX), (12, 0)] {
        let mut corrupt = original.clone();
        corrupt[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        std::fs::write(&path, corrupt).unwrap();
        assert!(QuantizedDataset::<L2U8>::load(&path).is_err());
    }
    let jpath = dir.join("jl");
    JLSparseDataset::<1024>::build_from(&index.dataset, 42)
        .save(&jpath)
        .unwrap();
    let mut bytes = std::fs::read(&jpath).unwrap();
    bytes[40..44].copy_from_slice(&33u32.to_le_bytes());
    std::fs::write(&jpath, bytes).unwrap();
    assert!(JLSparseDataset::<1024>::load(&jpath).is_err());
    std::fs::remove_dir_all(dir).unwrap();
}
