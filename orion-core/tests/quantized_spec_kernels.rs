/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_core::model::dataset::{L2U8, L2U16, MipsI8, MipsI16, QuantSpec};

fn check<Q: QuantSpec>()
where
    Q::Storage: Into<f64>,
{
    let values: [f32; 128] = std::array::from_fn(|i| (i % 17) as f32 - 8.);
    let params = Q::build_params(&values);
    Q::validate_params(&params).unwrap();
    let query: [Q::Storage; 32] = std::array::from_fn(|i| Q::quantize_scalar(&params, values[i]));
    let rows: [[Q::Storage; 32]; 4] = std::array::from_fn(|row| {
        std::array::from_fn(|i| Q::quantize_scalar(&params, values[row * 32 + i]))
    });
    let expected: [f32; 4] = std::array::from_fn(|r| {
        rows[r]
            .iter()
            .zip(query)
            .map(|(&a, b)| {
                let (a, b) = (Into::<f64>::into(a), Into::<f64>::into(b));
                if Q::METRIC == orion_ann_core::CalibrationMetric::SquaredL2 {
                    (a - b).powi(2)
                } else {
                    -a * b
                }
            })
            .sum::<f64>() as f32
    });
    // SAFETY: each input is a fully initialized array with exactly 32 elements.
    let batch = unsafe {
        Q::distance_batch4::<32>(
            rows[0].as_ptr(),
            rows[1].as_ptr(),
            rows[2].as_ptr(),
            rows[3].as_ptr(),
            query.as_ptr(),
        )
    };
    for i in 0..4 {
        let single = unsafe { Q::distance::<32>(rows[i].as_ptr(), query.as_ptr()) };
        assert!((single - expected[i]).abs() <= expected[i].abs().max(1.) * 2e-6);
        assert!((batch[i] - expected[i]).abs() <= expected[i].abs().max(1.) * 2e-6);
    }
    let a = std::array::from_fn(|i| values[i]);
    let b = std::array::from_fn(|i| values[32 + i]);
    let truth = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            if Q::METRIC == orion_ann_core::CalibrationMetric::SquaredL2 {
                (a - b).powi(2)
            } else {
                -a * b
            }
        })
        .sum::<f32>();
    assert_eq!(Q::truth_distance::<32>(&a, &b), truth);
    assert!(Q::distance_scale_sq(&params).is_finite());
    assert!(Q::distance_scale_sq(&params) > 0.);
    let zero = Q::build_params(&[0.; 32]);
    Q::validate_params(&zero).unwrap();
}
#[test]
fn l2_u8_batch_and_truth_match_scalar() {
    check::<L2U8>();
}
#[test]
fn l2_u16_batch_and_truth_match_scalar() {
    check::<L2U16>();
}
#[test]
fn mips_i8_batch_and_truth_match_scalar() {
    check::<MipsI8>();
}
#[test]
fn mips_i16_batch_and_truth_match_scalar() {
    check::<MipsI16>();
}
