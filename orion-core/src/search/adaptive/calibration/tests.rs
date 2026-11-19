/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Regression coverage for query preparation and the pre-refactor arithmetic.

use super::{
    query::{CalibrationQuery, prepare},
    scoring::{CalibrationScalar, wide_distance},
};
use crate::{CalibrationMetric, search::query::QueryInput};

/// Freezes the byte L2 path's float accumulation and checks scratch reuse across dimensions.
#[test]
fn byte_l2_retains_float_accumulation() {
    for dim in [1, 25, 32, 33, 128, 960, 1025, 4097] {
        let row: Vec<u8> = (0..dim).map(|i| ((i * 71 + 13) % 256) as u8).collect();
        let query: Vec<f32> = (0..dim).map(|i| ((i * 19 + 247) % 256) as f32).collect();
        let widened: Vec<f32> = row.iter().map(|&x| f32::from(x)).collect();
        let expected = orion_ann_vector::distance_l2_slice_f32(&query, &widened);
        let mut scratch = vec![f32::NAN; dim];
        let actual =
            u8::compatible_distance(CalibrationMetric::SquaredL2, &query, &row, &mut scratch)
                .unwrap();
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(scratch, widened);
    }
}

/// Checks integer dot/norm arithmetic separately from the widened L2 path.
#[test]
fn byte_ip_and_cosine_preserve_integer_reference() {
    let row = [255u8, 1, 0, 128];
    let query = [254., 2., 3., 127.];
    let mut scratch = [f32::NAN; 4];
    let ip = u8::compatible_distance(CalibrationMetric::InnerProduct, &query, &row, &mut scratch)
        .unwrap();
    assert_eq!(ip.to_bits(), (-81028.0f32).to_bits());
    let expected = (1.0 - 81028.0f64 / (80658.0f64.sqrt() * 81410.0f64.sqrt())) as f32;
    let cosine =
        u8::compatible_distance(CalibrationMetric::Cosine, &query, &row, &mut scratch).unwrap();
    assert_eq!(cosine.to_bits(), expected.to_bits());
    assert!(scratch.iter().all(|x| x.is_nan()));
    assert!(
        u8::compatible_distance(CalibrationMetric::Cosine, &[0.; 4], &row, &mut scratch).is_err()
    );
    assert!(
        u8::compatible_distance(CalibrationMetric::Cosine, &query, &[0; 4], &mut scratch).is_err()
    );
}

/// Protects the odd-dimension scalar IP order against accidental vector reduction changes.
#[test]
fn float_ip_keeps_odd_dimension_reduction() {
    let q = [1e20f32, 1., -1e20, 3., 4.];
    let row = [1.; 5];
    let expected = -q.iter().zip(row).map(|(q, x)| q * x).sum::<f32>();
    let actual =
        f32::compatible_distance(CalibrationMetric::InnerProduct, &q, &row, &mut [0.; 5]).unwrap();
    assert_eq!(actual.to_bits(), expected.to_bits());
}

/// A byte base must not truncate negative, fractional or wide query coordinates.
#[test]
fn preparation_selects_arithmetic_per_row() {
    let values = [1., 255., -1., 0.5];
    let rows = prepare::<u8>(QueryInput::F32(&values), 2, CalibrationMetric::SquaredL2).unwrap();
    assert!(matches!(&rows[0], CalibrationQuery::Compatible(v) if v.as_ref() == [1., 255.]));
    assert!(matches!(&rows[1], CalibrationQuery::Wide(v) if v == &[-1., 0.5]));
    for input in [
        QueryInput::I8(&[-2, 3]),
        QueryInput::U16(&[256, 65535]),
        QueryInput::U32(&[u32::MAX, 16_777_217]),
    ] {
        let rows = prepare::<u8>(input, 2, CalibrationMetric::SquaredL2).unwrap();
        assert!(matches!(rows[0], CalibrationQuery::Wide(_)));
    }
    let q = [u32::MAX, 16_777_217];
    let rows = prepare::<i32>(QueryInput::U32(&q), 2, CalibrationMetric::SquaredL2).unwrap();
    assert!(matches!(&rows[0], CalibrationQuery::Wide(v) if v == &[4294967295., 16777217.]));
}

/// Normalization precedes compatibility selection, and invalid inputs never reach scoring.
#[test]
fn preparation_normalizes_and_rejects_invalid_batches() {
    let q = [3u8, 4];
    let rows = prepare::<u8>(
        QueryInput::U8(&q).with_unit_norm(),
        2,
        CalibrationMetric::Cosine,
    )
    .unwrap();
    assert!(matches!(&rows[0], CalibrationQuery::Wide(v) if v == &[0.6, 0.8]));
    for q in [
        &[][..],
        &[1.][..],
        &[f32::NAN, 1.][..],
        &[f32::INFINITY, 1.][..],
    ] {
        assert!(prepare::<f32>(QueryInput::F32(q), 2, CalibrationMetric::SquaredL2).is_err());
    }
    assert!(prepare::<u8>(QueryInput::U8(&[0, 0]), 2, CalibrationMetric::Cosine).is_err());
}

/// Signed rows retain their sign and i32 coordinates beyond f32's exact integer range.
#[test]
fn signed_calibration_does_not_narrow_base_coordinates() {
    let q = [16_777_216f32, -16_777_216.];
    let row = [16_777_217i32, -16_777_217];
    let distance =
        i32::compatible_distance(CalibrationMetric::SquaredL2, &q, &row, &mut [0.; 2]).unwrap();
    assert_eq!(distance, 2.);
    assert_eq!(
        i8::compatible_distance(
            CalibrationMetric::InnerProduct,
            &[-2., 3.],
            &[-4, 5],
            &mut [0.; 2]
        )
        .unwrap(),
        -23.
    );
    assert_eq!(
        i16::compatible_distance(
            CalibrationMetric::SquaredL2,
            &[0., 0.],
            &[i16::MIN, i16::MAX],
            &mut [0.; 2]
        )
        .unwrap(),
        (32768f64.powi(2) + 32767f64.powi(2)) as f32
    );
    assert_eq!(
        wide_distance(CalibrationMetric::Cosine, &[1.], &[-1i32]).unwrap(),
        2.
    );
    assert!(wide_distance(CalibrationMetric::Cosine, &[1.], &[0i32]).is_err());
}
