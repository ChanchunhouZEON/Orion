/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT license.
 */

#![warn(missing_docs)]
#![allow(dead_code)]

//! Disk graph storage

use std::sync::Arc;

use crate::common::ANNResult;
use ::platform::{UnixAlignedFileReader, UnixIOContext};

/// Graph storage for disk index
/// One thread has one storage instance
pub struct DiskGraphStorage {
    /// Disk graph reader
    disk_graph_reader: Arc<UnixAlignedFileReader>,

    /// IOContext of current thread
    ctx: Arc<UnixIOContext>,
}

impl DiskGraphStorage {
    /// Create a new DiskGraphStorage instance
    pub fn new(disk_graph_reader: Arc<UnixAlignedFileReader>) -> ANNResult<Self> {
        let ctx = disk_graph_reader.get_ctx();
        Ok(Self {
            disk_graph_reader,
            ctx,
        })
    }

    // Read disk graph data
    // pub fn read<T>(&self, read_requests: &mut [AlignedRead<T>]) -> ANNResult<()> {}
}
