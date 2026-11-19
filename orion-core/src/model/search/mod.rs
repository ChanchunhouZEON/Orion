/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Query workspaces, visited membership and result payloads.
//!
//! These models own reusable search state or returned data. Traversal, cascade
//! scheduling and calibration algorithms remain in the crate's search module.

pub mod scratch;
pub(crate) mod search_result;
pub mod visited_set;

pub use scratch::{InMemScratchGuard, InMemScratchPool, InMemSearchScratch};
