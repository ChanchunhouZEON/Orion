/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Experimental paged graph search with bounded asynchronous object reads.
//!
//! Readers implement the runtime-independent platform contract. This baseline
//! uses bounded blocking positioned reads driven by Tokio; it makes no direct
//! I/O or physical-disk-latency claims. It does not authenticate object contents.
//! See [`FileReader`] for the immutable-generation and cancellation contract.
//! The search integration is graph-only: base and cascade sidecars still use
//! synchronous resident/mapped access. This crate does not automatically move
//! runtime objects to SSD or make mapped page faults asynchronous.

#![cfg(not(target_arch = "wasm32"))]
#![warn(missing_debug_implementations)]
#![doc = include_str!("../README.md")]

mod reader;
pub use reader::{FileReader, FileReaderOptions};

mod cache;
pub use cache::{CachedReader, PageCacheOptions, PageCacheStats};

/// Shared suspendable traversal with graph pages supplied by any ObjectReader.
/// Compose `SearchDriver::open(CachedReader::new(reader, options)?, entry).await`.
/// Base/sidecars remain available to the unchanged synchronous compute sessions.
pub use orion_ann_core::search::storage::SearchDriver;
