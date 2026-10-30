use diskann::model::PaddedDataset;
use vector::{
    DistanceFn, IpF32Distance, IpI16Distance, IpI8Distance, L2F32Distance, L2U16Distance,
    L2U8Distance, PaddedLayout,
};

fn check<K: DistanceFn>(
    make: impl Fn(usize) -> K::Storage,
    scalar: impl Fn(K::Storage, K::Storage) -> f64,
) where
    K::Storage: bytemuck::Pod + PartialEq + std::fmt::Debug,
{
    for dim in [
        1, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 255, 256, 257, 767, 768, 769,
        960, 1025,
    ] {
        let layout = PaddedLayout::for_kernel::<K>(dim).unwrap();
        let packed: Vec<_> = (0..5 * dim).map(&make).collect();
        let base = PaddedDataset::from_packed(&packed, layout).unwrap();
        let mut query = PaddedDataset::new(2, layout).unwrap();
        assert_eq!(base.as_slice().as_ptr() as usize % layout.alignment(), 0);
        for row in base.as_slice().chunks_exact(layout.stride_elements()) {
            assert!(row[dim..]
                .iter()
                .all(|x| *x == bytemuck::Zeroable::zeroed()));
        }
        // Reuse scratch; also address the final row with no allocation tail.
        for qid in [0, 1, 0] {
            query.set_row(qid, base.row(qid).unwrap()).unwrap();
            for ids in [vec![], vec![4], vec![4, 0, 3, 4, 2, 1]] {
                for near in [1, 8, 32, 64] {
                    let mut actual = vec![];
                    base.stream::<K>(&query, qid, &ids, near)
                        .unwrap()
                        .run(|id, d| actual.push((id, d)));
                    // Same validated inputs must preserve ordering and exact
                    // distances through the graph-search fast path.
                    let mut unchecked = vec![];
                    unsafe { base.stream_unchecked::<K>(&query, qid, &ids, near) }
                        .run(|id, d| unchecked.push((id, d)));
                    assert_eq!(actual, unchecked);
                    assert_eq!(actual.len(), ids.len());
                    for (&id, &(actual_id, distance)) in ids.iter().zip(&actual) {
                        assert_eq!(id, actual_id);
                        let expected: f64 = base
                            .row(id as usize)
                            .unwrap()
                            .iter()
                            .zip(query.row(qid).unwrap())
                            .map(|(&a, &b)| scalar(a, b))
                            .sum();
                        assert!(
                            (distance as f64 - expected).abs() <= expected.abs().max(1.) * 2e-5,
                            "dim={dim}, id={id}: {distance} != {expected}"
                        );
                        assert_eq!(distance, base.distance::<K>(id, &query, qid).unwrap());
                    }
                }
            }
        }
    }
}

#[test]
fn padded_numeric_kernels_match_logical_coordinates() {
    check::<L2F32Distance>(
        |i| (i % 31) as f32 * 0.125 - 2.,
        |a, b| (f64::from(a) - f64::from(b)).powi(2),
    );
    check::<IpF32Distance>(
        |i| (i % 31) as f32 * 0.125 - 2.,
        |a, b| -f64::from(a) * f64::from(b),
    );
    check::<L2U8Distance>(
        |i| (i % 256) as u8,
        |a, b| (f64::from(a) - f64::from(b)).powi(2),
    );
    check::<L2U16Distance>(
        |i| (i * 113 % 65536) as u16,
        |a, b| (f64::from(a) - f64::from(b)).powi(2),
    );
    check::<IpI8Distance>(
        |i| (i % 256) as u8 as i8,
        |a, b| -f64::from(a) * f64::from(b),
    );
    check::<IpI16Distance>(
        |i| (i * 113 % 65536) as u16 as i16,
        |a, b| -f64::from(a) * f64::from(b),
    );
}

#[test]
fn validates_layout_budget_and_search_inputs() {
    assert!(PaddedLayout::new::<u8>(0, 32).is_err());
    assert!(PaddedLayout::new::<()>(1, 32).is_err());
    assert!(PaddedLayout::new::<f32>(usize::MAX, 64).is_err());
    assert!(PaddedLayout::new::<f32>(3, 24).is_err());
    let layout = PaddedLayout::new::<f32>(100, 64).unwrap();
    assert_eq!(layout.stride_elements(), 112);
    assert_eq!(layout.padding_bytes(10).unwrap(), 480);
    assert_eq!(layout.allocation_bytes(10).unwrap(), 4480);
    assert!(layout.allocation_bytes(usize::MAX).is_err());
    assert!(PaddedDataset::<u8>::new(2, layout).is_err());
    assert!(PaddedDataset::<f32>::from_packed(&[0.; 99], layout).is_err());
    let mut base = PaddedDataset::<f32>::new(2, layout).unwrap();
    assert!(base.packed_mut().is_none());
    assert!(base.set_row(0, &[0.; 99]).is_err());
    assert!(base.stream::<L2F32Distance>(&base, 2, &[0], 32).is_err());
    assert!(base.stream::<L2F32Distance>(&base, 0, &[2], 32).is_err());
    let other =
        PaddedDataset::new(1, PaddedLayout::for_kernel::<L2F32Distance>(99).unwrap()).unwrap();
    assert!(base.stream::<L2F32Distance>(&other, 0, &[0], 32).is_err());
    assert!(PaddedLayout::for_kernel::<L2U8Distance>(L2U8Distance::MAX_COORDINATES + 1).is_err());
    let huge = PaddedDataset::<u8>::new(
        1,
        PaddedLayout::new::<u8>(L2U8Distance::MAX_COORDINATES + 1, L2U8Distance::CHUNK_BYTES)
            .unwrap(),
    )
    .unwrap();
    assert!(huge.distance::<L2U8Distance>(0, &huge, 0).is_err());
}

#[test]
fn integer_extremes_and_final_chunk_are_exact() {
    // AVX-512 u8 reduces signed machine lanes; the mathematical distance
    // above i32::MAX must still be interpreted as an unsigned sum.
    let dim = 40_000;
    let layout = PaddedLayout::for_kernel::<L2U8Distance>(dim).unwrap();
    let base = PaddedDataset::from_packed(&vec![255u8; dim], layout).unwrap();
    let query = PaddedDataset::from_packed(&vec![0u8; dim], layout).unwrap();
    assert_eq!(
        base.distance::<L2U8Distance>(0, &query, 0).unwrap(),
        (dim as u64 * 255 * 255) as f32
    );

    for dim in [1, 16, 31, 32, 33, 64, 65] {
        let layout = PaddedLayout::for_kernel::<L2U16Distance>(dim).unwrap();
        let base = PaddedDataset::from_packed(&vec![u16::MAX; dim], layout).unwrap();
        let query = PaddedDataset::from_packed(&vec![0u16; dim], layout).unwrap();
        assert_eq!(
            base.distance::<L2U16Distance>(0, &query, 0).unwrap(),
            (dim as u64 * 65535 * 65535) as f32
        );

        let layout = PaddedLayout::for_kernel::<IpI16Distance>(dim).unwrap();
        let base = PaddedDataset::from_packed(&vec![i16::MIN; dim], layout).unwrap();
        let query = PaddedDataset::from_packed(&vec![i16::MIN; dim], layout).unwrap();
        assert_eq!(
            base.distance::<IpI16Distance>(0, &query, 0).unwrap(),
            -((dim as u64 * 32768 * 32768) as f32)
        );
    }
}
