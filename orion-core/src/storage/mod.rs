/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

#[cfg(feature = "benchmark")]
pub mod mmap_storage;
#[cfg(feature = "benchmark")]
pub mod sector_layout;

#[cfg(feature = "benchmark")]
pub use mmap_storage::MmapStorage;
#[cfg(feature = "benchmark")]
pub use sector_layout::SectorLayout;
