/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#![cfg_attr(
    not(test),
    warn(clippy::panic, clippy::unwrap_used, clippy::expect_used)
)]

pub mod aligned_io;
pub mod aligned_reader;
pub mod file_handle;
pub mod graph_mmap;
pub mod io_context;
pub mod mmap_search;

pub use aligned_io::AlignedRead;
pub use aligned_reader::UnixAlignedFileReader;
pub use file_handle::{AccessMode, UnixFileHandle};
pub use graph_mmap::{AlgorithmId, GraphHeader, GraphWriter, MmapGraph};
pub use io_context::{Status, UnixIOContext};
pub use mmap_search::{mmap_greedy_search, mmap_greedy_search_with_data, MmapNeighbor};
