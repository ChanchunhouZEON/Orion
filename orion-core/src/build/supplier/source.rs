/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Sequential sources and borrowed partition rows.
use crate::Result;

/// Borrowed adjacency for one node. Local and remote are navigation edges;
/// extra is the supplier's discarded-neighbor partition, used by Orion's
/// existing search stages. All three lists must be mutually disjoint.
#[derive(Clone, Copy, Debug)]
pub struct NodePartitions<'a> {
    /// Confirmed navigation neighbors in supplier order.
    pub local: &'a [u32],

    /// Long-range navigation neighbors in supplier order.
    pub remote: &'a [u32],

    /// Retained non-navigation candidates available to later search stages.
    pub extra: &'a [u32],
}

/// Sequential source: return nodes 0..points, then `None` exactly at EOF.
/// Borrowed rows permit reusing a bounded decode buffer. Errors abort import;
/// partial indexes are dropped and never exposed to the caller.
pub trait PartitionSource {
    /// Borrows the next local/remote/extra partition in node order, or returns `None` at end of input.
    ///
    /// The borrow ends before the next mutable call so implementations may reuse row buffers.
    fn next_partition(&mut self) -> Result<Option<NodePartitions<'_>>>;
}

/// Borrow existing in-memory partitions without duplicating the full graph.
#[derive(Debug)]
pub struct SlicePartitions<'a> {
    rows: std::slice::Iter<'a, (Vec<u32>, Vec<u32>, Vec<u32>)>,
}
impl<'a> SlicePartitions<'a> {
    /// Adapts borrowed in-memory partitions to the sequential supplier interface without copying.
    pub fn new(rows: &'a [(Vec<u32>, Vec<u32>, Vec<u32>)]) -> Self {
        Self { rows: rows.iter() }
    }
}
impl PartitionSource for SlicePartitions<'_> {
    /// Lends the next partition in node order until the following mutable source call.
    fn next_partition(&mut self) -> Result<Option<NodePartitions<'_>>> {
        Ok(self
            .rows
            .next()
            .map(|(local, remote, extra)| NodePartitions {
                local,
                remote,
                extra,
            }))
    }
}

/// Move an already materialized graph into the runtime import worker. Only the
/// current row is retained after advancing; consumed rows are released early.
#[derive(Debug)]
pub struct OwnedPartitions {
    rows: std::vec::IntoIter<(Vec<u32>, Vec<u32>, Vec<u32>)>,
    current: Option<(Vec<u32>, Vec<u32>, Vec<u32>)>,
}
impl OwnedPartitions {
    /// Adapts owned partitions while retaining the current row long enough to lend it to the importer.
    pub fn new(rows: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>) -> Self {
        Self {
            rows: rows.into_iter(),
            current: None,
        }
    }
}
impl PartitionSource for OwnedPartitions {
    /// Lends the next partition in node order until the following mutable source call.
    fn next_partition(&mut self) -> Result<Option<NodePartitions<'_>>> {
        self.current = self.rows.next();
        Ok(self
            .current
            .as_ref()
            .map(|(local, remote, extra)| NodePartitions {
                local,
                remote,
                extra,
            }))
    }
}
