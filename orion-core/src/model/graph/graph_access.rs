/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Internal graph access shared by slab-backed and object-backed models.
//! Search consumes this contract without owning either model's storage policy.

use super::PhasedGraph;
use crate::Result;
use std::future::{Future, Ready, ready};

/// Partition slices whose backing is held for the duration of one expansion.
/// Concrete views decide whether that owner is a slab borrow or a read lease.
pub(crate) trait NodeView {
    /// Local then remote IDs in the stored traversal order; excludes extras.
    fn neighbors(&self) -> &[u32];

    /// Local and extra partitions used by the final precision pass.
    fn rerank_candidates(&self) -> (&[u32], &[u32]);
}

/// The traversal's storage boundary: acquire nodes without changing their
/// partition semantics. Resident acquisition is ready on the first poll;
/// object-backed acquisition can suspend before returning a validated view.
pub(crate) trait GraphAccess: Sync {
    /// A view retaining the node's backing while borrowed slices are consumed.
    type Node<'a>: NodeView
    where
        Self: 'a;
    /// Query-local acquisition state; it must not hold unrelated node locks.
    type Read<'a>: Future<Output = Result<Self::Node<'a>>> + Send
    where
        Self: 'a;
    /// Acquire a graph ID resolved by the traversal, not a file byte offset.
    fn node(&self, id: usize) -> Self::Read<'_>;

    /// Optional CPU cache hint. Absence of this hint never triggers an I/O fetch
    /// or changes traversal order; explicit prefetching needs a separate policy.
    fn prefetch_node(&self, _id: usize) {}
}

/// Borrowed node partitions from directly addressable graph storage.
pub(crate) struct ResidentNode<'a> {
    graph: &'a PhasedGraph,
    id: usize,
}

impl NodeView for ResidentNode<'_> {
    /// Borrows the navigation neighbors protected by this node view.
    #[inline(always)]
    fn neighbors(&self) -> &[u32] {
        self.graph.neighbors(self.id)
    }

    /// Borrows local and extra candidates without including remote navigation neighbors.
    #[inline(always)]
    fn rerank_candidates(&self) -> (&[u32], &[u32]) {
        self.graph.rerank_candidates(self.id)
    }
}

impl GraphAccess for PhasedGraph {
    type Node<'a> = ResidentNode<'a>;
    type Read<'a> = Ready<Result<ResidentNode<'a>>>;

    /// Returns an immediately ready borrowed node view without scheduling an I/O operation.
    #[inline(always)]
    fn node(&self, id: usize) -> Self::Read<'_> {
        ready(Ok(ResidentNode { graph: self, id }))
    }

    /// Issues a best-effort hint for a future node access without changing search state.
    #[inline(always)]
    fn prefetch_node(&self, id: usize) {
        PhasedGraph::prefetch_node(self, id);
    }
}
