//! Exact calibration kernels accept logical slices, including SIMD tails.
#[test]
fn runtime_exact_distances_cover_unaligned_rows_and_tails() {
    for dim in [
        1, 3, 4, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 128, 129, 960,
    ] {
        let a: Vec<f32> = (0..dim + 1)
            .map(|i| ((i * 31) % 107) as f32 / 17. - 3.)
            .collect();
        let b: Vec<f32> = (0..dim + 1)
            .map(|i| ((i * 13) % 101) as f32 / 23. - 2.)
            .collect();
        let a = &a[1..];
        let b = &b[1..];
        let l2 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>() as f32;
        let ip = -a
            .iter()
            .zip(b)
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum::<f64>() as f32;
        assert!(
            (vector::distance_l2_slice_f32(a, b) - l2).abs() <= l2.abs().max(1.) * 3e-6,
            "L2 dim={dim}"
        );
        assert!(
            (vector::distance_ip_slice_f32(a, b) - ip).abs() <= ip.abs().max(1.) * 3e-6,
            "IP dim={dim}"
        );
    }
}
#[test]
#[should_panic]
fn mismatched_exact_inputs_are_rejected() {
    vector::distance_l2_slice_f32(&[1., 2.], &[1.]);
}
