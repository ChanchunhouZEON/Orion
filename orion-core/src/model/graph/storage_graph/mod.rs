/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! A validated graph representation backed by immutable object reads.
//! This module owns record layout and node leases, not query execution.
//!
//! Native layout (u32 words, matching the existing PhasedGraph codec):
//! ```text
//! header: [nodes, max_degree, stride_words, base_local_count]
//! node:   [degree, extra_count, local_count, reserved=0,
//!          local IDs, remote IDs, extra IDs, unused padding]
//! offset: 16 + node_id * stride_words * 4
//! ```
//! The format adapter requests a whole node record and retains its lease while
//! partition slices are in use. It does not choose page size or cache policy.

use super::graph_access::{GraphAccess, NodeView};
use crate::{Error, Result};
use orion_ann_platform::reading::{ObjectReader, ReadLease, ReadRequest};
use std::{future::Future, pin::Pin};

/// Validated native graph header plus an immutable object reader, no full slab.
/// Every acquired record is validated before IDs reach unsafe distance streams.
/// Size/count validation is not file authentication; supply a verified immutable
/// generation. Node IDs and partition order are preserved exactly.
#[derive(Debug)]
pub struct StorageGraph<R> {
    // Owns the immutable generation; graph views never outlive its read leases.
    reader: R,
    nodes: usize,
    // Navigation start belongs to generation metadata, not the slab header.
    entry: u32,
    // Includes the record header, both neighbor partitions, extras and padding.
    stride_words: usize,
    max_degree: usize,
}

impl<R: ObjectReader> StorageGraph<R> {
    /// Read and validate the slab header without loading all graph records.
    ///
    /// `entry` comes from the same verified generation's metadata. This checks
    /// object length, stride and header bounds; individual records are checked
    /// on acquisition before any neighbor IDs reach distance kernels. It does
    /// not authenticate content or prove whole-graph connectivity/uniqueness.
    ///
    /// # Errors
    /// Propagates header I/O and typed-view errors; rejects invalid counts,
    /// entry points, stride or payload length. An invalid later record is
    /// reported by the search that first acquires it.
    pub async fn open(reader: R, entry: u32) -> Result<Self> {
        let lease = reader.read(ReadRequest::new(0, 16)?).await?;
        let header = lease.as_slice::<u32>()?;
        let [nodes, degree, stride, local] = <[u32; 4]>::try_from(header)
            .map_err(|_| Error::invalid_data("invalid graph header length"))?;
        let expected = (nodes as u64)
            .checked_mul(stride as u64)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(16));
        if nodes == 0
            || nodes == u32::MAX
            || entry >= nodes
            || degree == 0
            || stride < 4
            || stride % 16 != 0
            || degree > stride - 4
            || local > degree
            || expected != Some(reader.byte_len())
        {
            return Err(Error::invalid_data(
                "invalid storage graph header or payload length",
            ));
        }
        drop(lease);
        Ok(Self {
            reader,
            nodes: nodes as usize,
            entry,
            stride_words: stride as usize,
            max_degree: degree as usize,
        })
    }

    /// Returns the validated entry point supplied by the same generation's metadata.
    pub(crate) fn entry_point(&self) -> u32 {
        self.entry
    }

    /// Returns the fixed node-record stride in u32 words, including metadata and padding.
    pub(crate) fn stride_words(&self) -> usize {
        self.stride_words
    }

    /// Declared node count after header/length validation.
    /// This does not imply that every record has already been read and checked.
    pub fn num_nodes(&self) -> usize {
        self.nodes
    }

    /// Inspect backend state, such as page-cache counters, without replacing
    /// the generation. Backend operations like clearing retained pages do not
    /// change this graph's layout or node IDs.
    pub fn reader(&self) -> &R {
        &self.reader
    }
}

// Keep the lease beside the validated partition boundaries. Returned slices
// borrow this node view, so a cache frame cannot be reclaimed mid-computation.
/// Validated node view whose read lease keeps storage alive during traversal.
pub(crate) struct StoredNode<'a> {
    lease: ReadLease<'a>,
    degree: usize,
    local: usize,
    extra: usize,
}

impl NodeView for StoredNode<'_> {
    /// Borrows the navigation neighbors protected by this node view.
    fn neighbors(&self) -> &[u32] {
        &self
            .lease
            .as_slice::<u32>()
            .expect("validated graph record")[4..4 + self.degree]
    }

    /// Borrows local and extra candidates without including remote navigation neighbors.
    fn rerank_candidates(&self) -> (&[u32], &[u32]) {
        let words = self
            .lease
            .as_slice::<u32>()
            .expect("validated graph record");
        (
            &words[4..4 + self.local],
            &words[4 + self.degree..4 + self.degree + self.extra],
        )
    }
}

impl<R: ObjectReader> GraphAccess for StorageGraph<R> {
    type Node<'a>
        = StoredNode<'a>
    where
        Self: 'a;
    type Read<'a>
        = Pin<Box<dyn Future<Output = Result<StoredNode<'a>>> + Send + 'a>>
    where
        Self: 'a;

    /// Reads and validates one fixed-stride node record, retaining its lease until the node view drops.
    fn node(&self, id: usize) -> Self::Read<'_> {
        Box::pin(async move {
            if id >= self.nodes {
                return Err(Error::invalid_data("node ID outside graph"));
            }
            let bytes = self
                .stride_words
                .checked_mul(4)
                .ok_or_else(|| Error::invalid_data("record byte overflow"))?;
            let offset = 16 + id as u64 * bytes as u64;
            let lease = self.reader.read(ReadRequest::new(offset, bytes)?).await?;
            let words = lease.as_slice::<u32>()?;
            if words.len() != self.stride_words {
                return Err(Error::invalid_data("short graph record"));
            }
            let degree = words[0] as usize;
            let extra = words[1] as usize;
            let local = words[2] as usize;
            let end = degree.checked_add(extra).and_then(|n| n.checked_add(4));
            if degree > self.max_degree
                || local > degree
                || words[3] != 0
                || end.is_none_or(|n| n > words.len())
            {
                return Err(Error::invalid_data("invalid graph record counts"));
            }
            if words[4..end.expect("checked counts")]
                .iter()
                .any(|&n| n as usize >= self.nodes)
            {
                return Err(Error::invalid_data(
                    "graph record contains out-of-range neighbor",
                ));
            }
            Ok(StoredNode {
                lease,
                degree,
                local,
                extra,
            })
        })
    }
}
