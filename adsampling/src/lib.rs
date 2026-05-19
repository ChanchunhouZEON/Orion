/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! ADSampling — random rotation + probabilistic early-abandon distance.
//!
//! The SIMD early-abort arithmetic lives in the `vector` crate (trait method
//! `FullPrecisionDistance::distance_compare_adsampling`). This crate provides
//! the rotation primitive: build a fixed random orthogonal matrix once, apply
//! it to data vectors and queries so the per-dimension variance is
//! approximately uniform — a pre-condition for the scaled partial-sum test.

pub mod rotator;

pub use rotator::Rotator;
