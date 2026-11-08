/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! ADSampling — random rotation + probabilistic early-abandon distance.
//!
//! Available only with the opt-in `benchmark` feature. Production builds do not
//! expose rotation or enable its random-number and parallelism dependencies.
//!
//! The SIMD early-abort arithmetic lives in `orion-ann-vector` (trait method
//! `FullPrecisionDistance::distance_compare_adsampling`). This crate provides
//! the rotation primitive: build a fixed random orthogonal matrix once, apply
//! it to data vectors and queries so the per-dimension variance is
//! approximately uniform — a pre-condition for the scaled partial-sum test.

#[cfg(feature = "benchmark")]
pub mod rotator;

#[cfg(feature = "benchmark")]
pub use rotator::Rotator;
