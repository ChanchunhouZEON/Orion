/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Adversarial tests for the safe sidecar boundary, separate from performance tests.
#[path = "support/sidecar_file.rs"]
mod sidecar_file;
use orion_ann_core::algorithm::search::stage::Admission;
use orion_ann_core::algorithm::search::stage::admission::L2U8Admission;
use orion_ann_core::model::dataset::{
    jl_hadamard_dataset::JlHadamardDataset,
    jl_sparse_dataset::{JLSparseDataset, JLSparseDatasetMips, hamming_dispatch, hamming_scalar},
    l2_kt_dataset::L2KTDataset,
    quantized_spec::{L2U8, L2U16, MipsI8, MipsI16, QuantParamsL2},
    rabitq_b4_dataset::RabitQ4Dataset,
    rabitq_dataset::{RabitQDataset, apply_rotation_pub},
    rows::F32Rows,
    scalar_dataset::ScalarQuantizedDataset,
};
use orion_ann_core::{ErrorCode, PaddedDataset, PaddedLayout};
use orion_ann_vector::L2F32Distance;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "orion-sidecar-safety-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn base() -> PaddedDataset<f32> {
    PaddedDataset::from_packed(
        &[0., 1., 2., 3., 4., 5.],
        PaddedLayout::for_kernel::<L2F32Distance>(3).unwrap(),
    )
    .unwrap()
}
fn rejects(action: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(action)).is_err());
}

// A scoped pool joins its workers before each test returns. This keeps Miri's
// thread-lifetime checks enabled, without relying on Rayon's process-global pool.
fn with_pool(action: impl FnOnce() + Send) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_scoped(|thread| thread.run(), |pool| pool.install(action))
        .unwrap();
}

#[test]
fn safe_row_entry_prefetch_and_rerank_reject_bad_ids() {
    // Load a tiny valid packed cache so the Miri boundary test does not depend
    // on Rayon's global pool or Crossbeam's intrusive-list borrow model.
    let temp = Temp::new();
    let path = temp.0.join("rows.qds");
    let mut bytes = Vec::new();
    for value in [0x5144_5300u32, 1, 2, 3, 51f32.to_bits(), 0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[0u8, 51, 102, 153, 204, 255]);
    std::fs::write(&path, bytes).unwrap();
    let q = sidecar_file::decode(path, ScalarQuantizedDataset::<L2U8>::decode).unwrap();
    let saved = temp.0.join("saved.qds");
    q.encode(&mut std::fs::File::create(&saved).unwrap())
        .unwrap();
    assert_eq!(
        sidecar_file::decode(saved, ScalarQuantizedDataset::<L2U8>::decode)
            .unwrap()
            .data(),
        q.data()
    );
    assert!(q.row(2).is_none());
    assert!(q.distance(2, &[0; 3]).is_err());
    assert!(q.distance(0, &[0; 2]).is_err());
    rejects(|| q.prefetch_vector(2));
    q.prefetch_vector(1);
    let admission = L2U8Admission::new(&q);
    let session = (&admission as &Admission).open(&[0., 1., 2.]);
    rejects(|| {
        session.entry_distance(2);
    });
}

#[test]
fn extreme_finite_queries_saturate_instead_of_overflowing() {
    with_pool(|| {
        let p = QuantParamsL2::from_range(-1., 1.);
        assert_eq!(p.quantize_scalar(1e30), 255);
        assert_eq!(p.quantize_scalar(-1e30), 0);
        let p = QuantParamsL2::from_range_u16(-1., 1.);
        assert_eq!(p.quantize_scalar_u16(1e30), u16::MAX);
        assert_eq!(p.quantize_scalar_u16(-1e30), 0);
        // Wide intermediate arithmetic must also work for extreme on-disk offsets.
        for offset in [i32::MIN, i32::MAX] {
            let p = QuantParamsL2 { slope: 1., offset };
            assert_eq!(p.quantize_scalar(1e30), 255);
            assert_eq!(p.quantize_scalar(-1e30), 0);
            assert_eq!(p.quantize_scalar_u16(1e30), u16::MAX);
            assert_eq!(p.quantize_scalar_u16(-1e30), 0);
        }
        let q = ScalarQuantizedDataset::<L2U8>::try_from_f32_dataset(&base()).unwrap();
        assert_eq!(
            q.try_quantize_query(&[f32::NAN; 3], true)
                .unwrap_err()
                .code(),
            ErrorCode::InvalidArgument
        );
        assert_eq!(
            q.try_quantize_query(&[0.; 2], false).unwrap_err().code(),
            ErrorCode::DimensionMismatch
        );
    });
}

struct BadRows {
    rows: usize,
    dim: usize,
    values: Vec<f32>,
}
impl F32Rows for BadRows {
    fn len(&self) -> usize {
        self.rows
    }
    fn dimension(&self) -> usize {
        self.dim
    }
    fn row(&self, _: usize) -> &[f32] {
        &self.values
    }
    // A custom provider cannot bypass validation with a forged range.
    fn range(&self) -> (f32, f32) {
        (0., 1.)
    }
}
#[test]
fn constructors_validate_provider_shape_and_values() {
    with_pool(|| {
        for data in [
            BadRows {
                rows: 0,
                dim: 3,
                values: vec![0.; 3],
            },
            BadRows {
                rows: 1,
                dim: 0,
                values: vec![],
            },
            BadRows {
                rows: usize::MAX,
                dim: 3,
                values: vec![0.; 3],
            },
            BadRows {
                rows: 1,
                dim: 3,
                values: vec![0.; 2],
            },
            BadRows {
                rows: 1,
                dim: 3,
                values: vec![f32::NAN; 3],
            },
        ] {
            assert!(ScalarQuantizedDataset::<L2U8>::try_from_f32_dataset(&data).is_err());
            assert!(L2KTDataset::try_build_from(&data).is_err());
            assert!(JLSparseDataset::<128>::try_build_from(&data, 42).is_err());
            assert!(JLSparseDatasetMips::<128>::try_build_from(&data, 42).is_err());
            assert!(JlHadamardDataset::try_build_from(&data, 42).is_err());
            assert!(RabitQDataset::try_build_from(&data, 42).is_err());
            assert!(RabitQ4Dataset::try_build_from(&data, 42).is_err());
        }
        assert!(JLSparseDataset::<7>::try_build_from(&base(), 42).is_err());
        assert!(JlHadamardDataset::try_build_with_padding(&base(), 42, 3).is_err());
    });
}

#[test]
fn safe_simd_helpers_validate_lengths_and_keep_tails() {
    rejects(|| {
        hamming_dispatch(&[0; 32], &[0; 1]);
    });
    rejects(|| {
        apply_rotation_pub(&[0.; 256], &[0.; 1], &mut [0.; 16]);
    });
    rejects(|| {
        apply_rotation_pub(&[0.; 1], &[0.; 16], &mut [0.; 16]);
    });
    for n in 0..130 {
        let p: Vec<_> = (0..n).map(|i| i as u8).collect();
        let q: Vec<_> = p.iter().map(|x| !x).collect();
        assert_eq!(hamming_dispatch(&p, &q), (8 * n) as u32);
        assert_eq!(hamming_dispatch(&p, &q), hamming_scalar(&p, &q));
    }
}

#[test]
fn all_codecs_reject_truncation_trailing_bytes_and_huge_row_headers() {
    with_pool(|| {
        let temp = Temp::new();
        let base = base();
        macro_rules! verify {
            ($ty:ty, $value:expr) => {{
                let path = temp.0.join("cache");
                let value = $value;
                value
                    .encode(&mut std::fs::File::create(&path).unwrap())
                    .unwrap();
                let bytes = std::fs::read(&path).unwrap();
                sidecar_file::decode(&path, <$ty>::decode).unwrap();
                for bad in [
                    bytes[..bytes.len() - 1].to_vec(),
                    {
                        let mut b = bytes.clone();
                        b.push(0);
                        b
                    },
                    {
                        let mut b = bytes.clone();
                        b[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
                        b
                    },
                ] {
                    std::fs::write(&path, bad).unwrap();
                    assert!(
                        sidecar_file::decode(&path, <$ty>::decode).is_err(),
                        "{} accepted malformed cache",
                        stringify!($ty)
                    );
                }
            }};
        }
        verify!(
            ScalarQuantizedDataset<L2U8>,
            ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&base)
        );
        verify!(
            ScalarQuantizedDataset<L2U16>,
            ScalarQuantizedDataset::<L2U16>::from_f32_dataset(&base)
        );
        verify!(
            ScalarQuantizedDataset<MipsI8>,
            ScalarQuantizedDataset::<MipsI8>::from_f32_dataset(&base)
        );
        verify!(
            ScalarQuantizedDataset<MipsI16>,
            ScalarQuantizedDataset::<MipsI16>::from_f32_dataset(&base)
        );
        verify!(L2KTDataset, L2KTDataset::build_from(&base));
        verify!(
            JLSparseDataset<128>,
            JLSparseDataset::<128>::build_from(&base, 42)
        );
        verify!(
            JLSparseDatasetMips<128>,
            JLSparseDatasetMips::<128>::build_from(&base, 42)
        );
        verify!(
            JlHadamardDataset,
            JlHadamardDataset::build_with_padding(&base, 42, 4)
        );
        verify!(RabitQDataset, RabitQDataset::build_from(&base, 42));
        verify!(RabitQ4Dataset, RabitQ4Dataset::build_from(&base, 42));
    });
}

#[test]
fn invalid_quantizer_parameters_are_rejected_on_load() {
    with_pool(|| {
        let temp = Temp::new();
        let path = temp.0.join("cache");
        ScalarQuantizedDataset::<L2U8>::from_f32_dataset(&base())
            .encode(&mut std::fs::File::create(&path).unwrap())
            .unwrap();
        let valid = std::fs::read(&path).unwrap();
        for scale in [
            0.,
            -1.,
            f32::NAN,
            f32::INFINITY,
            f32::MAX,
            f32::MIN_POSITIVE,
        ] {
            let mut bad = valid.clone();
            bad[16..20].copy_from_slice(&scale.to_le_bytes());
            std::fs::write(&path, bad).unwrap();
            assert_eq!(
                sidecar_file::decode(&path, ScalarQuantizedDataset::<L2U8>::decode)
                    .unwrap_err()
                    .code(),
                ErrorCode::InvalidData
            );
        }
    });
}
