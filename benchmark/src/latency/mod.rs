/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Closed-loop, single-query request latency through the public Index API.
//! This unpublished benchmark workspace owns all clocks and distributions;
//! production runtime/core and their SIMD/prefetch paths remain unchanged.
pub mod config;
pub mod evaluation;
pub mod report;
pub mod runner;
pub mod statistics;
