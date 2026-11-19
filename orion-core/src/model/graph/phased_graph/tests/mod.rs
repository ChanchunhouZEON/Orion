/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Internal graph contracts; no public low-level graph API is needed for testing.
mod adversarial;
mod concurrency;
#[cfg(feature = "parlay")]
mod parlay_memory;
mod randomized;
mod support;

mod mapped;
