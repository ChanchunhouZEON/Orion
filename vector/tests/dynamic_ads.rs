use vector::{FullPrecisionDistance, distance_l2_adsampling_slice_f32};

fn compare<const N: usize>() {
    let a = std::array::from_fn::<_, N, _>(|i| (i % 19) as f32 * 0.5);
    let b = std::array::from_fn::<_, N, _>(|i| (i % 13) as f32 * 0.25);
    for bound in [0., 1., 100., 1000., f32::MAX] {
        let typed = <[f32; N] as FullPrecisionDistance<f32, N>>::distance_compare_adsampling(
            &a, &b, bound, 2.1,
        );
        let dynamic = distance_l2_adsampling_slice_f32(&a, &b, bound, 2.1);
        assert_eq!(typed, dynamic, "dimension={N}, bound={bound}");
    }
}
#[test]
fn ads_preserves_checkpoint_and_reduction_order() {
    compare::<32>();
    compare::<100>();
    compare::<128>();
    compare::<960>();
}
#[test]
fn ads_handles_unaligned_logical_tails() {
    for n in [1, 3, 17, 31, 33, 65, 129, 961] {
        let a = vec![1.; n + 1];
        let b = vec![3.; n + 1];
        assert_eq!(
            distance_l2_adsampling_slice_f32(&a[1..], &b[1..], f32::MAX, 2.1),
            4. * n as f32
        );
        assert!(distance_l2_adsampling_slice_f32(&a[1..], &b[1..], 1., 2.1) < 0. || n < 64);
    }
}
