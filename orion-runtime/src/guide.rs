/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Start with the [end-to-end example](quick_start) to build your first index.
//! For a specific task, choose a guide below.
//!
//! | Task | Guide |
//! | --- | --- |
//! | Select a metric, graph source or memory budget | [Configuration](configuration) |
//! | Index a file of byte vectors or import an existing STAG graph | [Construction](build) |
//! | Search embeddings in batches and calibrate for multiple values of `k` | [Search](search) |
//! | Restart a service with saved calibration and a different worker count | [Working caches and snapshots](cache) |
#[doc = include_str!("guide/configuration.md")]
pub mod configuration {}
#[doc = include_str!("guide/build.md")]
pub mod build {}
#[doc = include_str!("guide/cache.md")]
pub mod cache {}
#[doc = include_str!("guide/search.md")]
pub mod search {}

/// End-to-end service usage from the package README.
#[doc = include_str!("../README.md")]
pub mod quick_start {}
