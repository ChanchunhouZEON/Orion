/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Coordinate validation and exact metric contracts for base representations.
use crate::{CalibrationMetric, Error, Result};
use bytemuck::Pod;
use orion_ann_vector::{DistanceKernelBackend, L2F32Distance, L2U8Distance};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u8 {}
}

pub trait DynamicElement: sealed::Sealed + Pod + Send + Sync + Into<f32> {
    type Kernel: DistanceKernelBackend<Storage = Self>;
    /// Layout bound may exceed the legacy L2 accumulator bound.
    type LayoutKernel: DistanceKernelBackend<Storage = Self>;
    fn valid_coordinate(value: Self) -> bool;
    fn valid_query(value: f32) -> bool {
        value.is_finite()
    }
    fn supports_calibration(metric: CalibrationMetric) -> bool;
    fn calibration_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        scratch: &mut [f32],
    ) -> Result<f32>;
}
impl DynamicElement for f32 {
    type Kernel = L2F32Distance;
    type LayoutKernel = L2F32Distance;
    fn supports_calibration(_: CalibrationMetric) -> bool {
        true
    }
    fn calibration_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        _: &mut [f32],
    ) -> Result<f32> {
        crate::search::adaptive::calibration::exact_distance(metric, query, row)
    }
    fn valid_coordinate(value: Self) -> bool {
        value.is_finite()
    }
}
impl DynamicElement for u8 {
    type Kernel = L2U8Distance;
    type LayoutKernel = orion_ann_vector::IpU8Distance;
    fn valid_query(v: f32) -> bool {
        v.is_finite() && (0.0..=255.0).contains(&v) && v.fract() == 0.0
    }
    fn supports_calibration(_: CalibrationMetric) -> bool {
        true
    }
    fn calibration_distance(
        metric: CalibrationMetric,
        query: &[f32],
        row: &[Self],
        scratch: &mut [f32],
    ) -> Result<f32> {
        if metric != CalibrationMetric::SquaredL2 {
            let mut dot = 0u64;
            let (mut qnorm, mut norm) = (0u64, 0u64);
            for (&q, &x) in query.iter().zip(row) {
                let (q, x) = (q as u64, u64::from(x));
                dot += q * x;
                if metric == CalibrationMetric::Cosine {
                    qnorm += q * q;
                    norm += x * x;
                }
            }
            return if metric == CalibrationMetric::InnerProduct {
                Ok(-(dot as f32))
            } else {
                if qnorm == 0 || norm == 0 {
                    return Err(Error::invalid_argument("cosine requires nonzero vectors"));
                }
                Ok(
                    (1.0 - dot as f64 / ((qnorm as f64).sqrt() * (norm as f64).sqrt()))
                        .clamp(0., 1.) as f32,
                )
            };
        }
        for (out, &v) in scratch.iter_mut().zip(row) {
            *out = f32::from(v);
        }
        crate::search::adaptive::calibration::exact_distance(metric, query, scratch)
    }
    fn valid_coordinate(_: Self) -> bool {
        true
    }
}
