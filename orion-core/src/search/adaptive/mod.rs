/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Adaptive search policy: calibration, convergence detection and early stopping.
//! These modules choose when traversal changes phase or stops; distance kernels
//! and prefetch scheduling remain in the ordinary search path.
pub mod calibrate;
pub mod convergence;
pub mod early_exit;

pub(crate) mod calibration;
