/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! **PhasedGraph** — cache-line aligned internal graph for two-phase
//! staged search.
//!
//! Per-node slot layout (fixed stride, cache-line aligned):
//! ```text
//! ┌─── Header (4 × u32 = 16 bytes) ─────────────────────────────────────────────────┐
//! │ degree (R) │ extra_count (E) │ local_count (L) │ reserved                       │
//! ├─── Data ────────────────────────────────────────────────────────────────────────┤
//! │ local_neighbors  [u32 × L]       ← navigation + reranking                       │
//! │ remote_neighbors [u32 × (R − L)] ← navigation only                              │
//! │ extra_candidates [u32 × E]       ← reranking only (after actual degree)         │
//! │ unused capacity + cache-line padding                                            │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! Backed by `AlignedBoxWithSlice<u32>`. Orion owns the public boundary: readers
//! borrow the continuous slab directly, while construction and mutation require
//! exclusive access. No read-side lock, per-node reader counter or pointer table
//! is needed. The graph type and mutable storage are not exported to callers.

use crate::{Error, Result};
use orion_ann_diskann::common::AlignedBoxWithSlice;
mod io;
#[cfg(test)]
mod tests;

const CACHE_LINE_BYTES: usize = 64;
const HEADER_U32: usize = 4;

#[derive(Debug)]
pub(crate) struct PhasedGraph {
    buffer: AlignedBoxWithSlice<u32>,
    num_nodes: usize,
    stride: usize,
    pub(crate) max_degree: u32,
    max_extra: usize,
    pub(crate) base_local_count: u32,
}
impl PhasedGraph {
    /// Allocate an empty graph. All records initially have zero edges.
    ///
    /// # Errors
    /// Rejects zero/overflowing shapes and reports allocation failure.
    #[cfg(test)]
    pub(crate) fn try_empty(num_nodes: usize, max_degree: u32, max_extra: usize) -> Result<Self> {
        Self::try_empty_with_storage(num_nodes, max_degree, max_extra, &Default::default())
    }

    pub(crate) fn try_empty_with_storage(
        num_nodes: usize,
        max_degree: u32,
        max_extra: usize,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        let stride = Self::checked_stride(num_nodes, max_degree, max_extra)?;
        Self::allocate_with_storage(num_nodes, max_degree, max_extra, stride, 0, storage)
    }

    pub(crate) fn placement(&self) -> orion_ann_platform::region::Placement {
        self.buffer.placement()
    }

    fn checked_stride(num_nodes: usize, max_degree: u32, max_extra: usize) -> Result<usize> {
        if num_nodes == 0 || num_nodes >= u32::MAX as usize || max_degree == 0 {
            return Err(Error::invalid_argument("invalid graph shape"));
        }
        let words = (max_degree as usize)
            .checked_add(max_extra)
            .and_then(|n| n.checked_add(HEADER_U32))
            .ok_or_else(|| Error::invalid_argument("graph capacity overflow"))?;
        let bytes = words
            .checked_mul(4)
            .and_then(|n| n.checked_add(CACHE_LINE_BYTES - 1))
            .ok_or_else(|| Error::invalid_argument("graph stride overflow"))?;
        let stride = bytes / CACHE_LINE_BYTES * (CACHE_LINE_BYTES / 4);
        if stride > u32::MAX as usize {
            return Err(Error::invalid_argument(
                "graph stride exceeds cache header range",
            ));
        }
        Ok(stride)
    }

    fn allocate(
        num_nodes: usize,
        max_degree: u32,
        max_extra: usize,
        stride: usize,
        base_local_count: u32,
    ) -> Result<Self> {
        Self::allocate_with_storage(
            num_nodes,
            max_degree,
            max_extra,
            stride,
            base_local_count,
            &Default::default(),
        )
    }

    fn allocate_with_storage(
        num_nodes: usize,
        max_degree: u32,
        max_extra: usize,
        stride: usize,
        base_local_count: u32,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        let total = num_nodes
            .checked_mul(stride)
            .ok_or_else(|| Error::invalid_argument("graph allocation overflow"))?;
        Ok(Self {
            buffer: AlignedBoxWithSlice::new_with_storage(total, CACHE_LINE_BYTES, storage)?,
            num_nodes,
            stride,
            max_degree,
            max_extra,
            base_local_count,
        })
    }

    /// Construct in the final slab. The callback may split exclusive rows among
    /// workers; it must join them before returning. Nothing is published on error.
    ///
    /// # Errors
    /// Rejects invalid counts, reserved words and neighbor IDs after the callback.
    #[cfg(test)]
    pub(crate) fn try_build_with(
        num_nodes: usize,
        max_degree: u32,
        max_extra: usize,
        build: impl FnOnce(&mut [u32], usize) -> Result<()>,
    ) -> Result<Self> {
        Self::try_build_with_storage(num_nodes, max_degree, max_extra, build, &Default::default())
    }

    #[cfg(any(feature = "parlay", test))]
    pub(crate) fn try_build_with_storage(
        num_nodes: usize,
        max_degree: u32,
        max_extra: usize,
        build: impl FnOnce(&mut [u32], usize) -> Result<()>,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        let mut graph = Self::try_empty_with_storage(num_nodes, max_degree, max_extra, storage)?;
        build(graph.buffer.as_mut_slice(), graph.stride)?;
        graph.validate_rows()?;
        Ok(graph)
    }

    fn validate_rows(&self) -> Result<()> {
        for (node, row) in self.buffer.chunks_exact(self.stride).enumerate() {
            let degree = row[0] as usize;
            let extra = row[1] as usize;
            let end = degree
                .checked_add(extra)
                .and_then(|n| n.checked_add(HEADER_U32));
            if degree > self.max_degree as usize
                || extra > self.max_extra
                || row[2] as usize > degree
                || row[3] != 0
                || end.is_none_or(|end| end > self.stride)
            {
                return Err(
                    Error::invalid_data("invalid graph record counts").with_detail("node", node)
                );
            }
            if row[HEADER_U32..end.unwrap()]
                .iter()
                .any(|&id| id as usize >= self.num_nodes)
            {
                return Err(Error::invalid_data("graph neighbor ID outside graph")
                    .with_detail("node", node));
            }
        }
        Ok(())
    }

    fn validate_partition(
        &self,
        node: usize,
        local: &[u32],
        remote: &[u32],
        extra: &[u32],
    ) -> Result<()> {
        if node >= self.num_nodes
            || local
                .len()
                .checked_add(remote.len())
                .is_none_or(|n| n > self.max_degree as usize)
            || extra.len() > self.max_extra
            || local
                .iter()
                .chain(remote)
                .chain(extra)
                .any(|&id| id as usize >= self.num_nodes)
        {
            return Err(Error::invalid_data(
                "graph partition exceeds capacity or contains invalid IDs",
            )
            .at("write_graph_node")
            .with_detail("node", node)
            .with_detail("nodes", self.num_nodes)
            .with_detail("max_degree", self.max_degree)
            .with_detail("max_extra", self.max_extra));
        }
        Ok(())
    }

    // Only validated callers reach this helper. Safe slicing also prevents a
    // future validation regression from becoming an out-of-bounds pointer write.
    fn write_row(row: &mut [u32], local: &[u32], remote: &[u32], extra: &[u32]) {
        let degree = local.len() + remote.len();
        row[..4].copy_from_slice(&[degree as u32, extra.len() as u32, local.len() as u32, 0]);
        let data = &mut row[HEADER_U32..];
        data[..local.len()].copy_from_slice(local);
        data[local.len()..degree].copy_from_slice(remote);
        data[degree..degree + extra.len()].copy_from_slice(extra);
    }

    /// Internal exclusive update; no reference into the slab may coexist with it.
    ///
    /// # Errors
    /// Rejects invalid IDs/capacities without changing the graph.
    // Reserved for audited exclusive updates; currently exercised by safety tests.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn set_partition(
        &mut self,
        node: usize,
        local: &[u32],
        remote: &[u32],
        extra: &[u32],
    ) -> Result<()> {
        self.validate_partition(node, local, remote, extra)?;
        let row = &mut self.buffer[node * self.stride..(node + 1) * self.stride];
        Self::write_row(row, local, remote, extra);
        Ok(())
    }

    /// Fallible parallel construction from already partitioned edges.
    ///
    /// # Errors
    /// Validates every partition before writing; rejects invalid IDs/capacities.
    pub(crate) fn try_from_partitions(
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        max_degree: u32,
        max_extra: usize,
    ) -> Result<Self> {
        Self::try_from_partitions_with_storage(
            partitions,
            max_degree,
            max_extra,
            &Default::default(),
        )
    }

    pub(crate) fn try_from_partitions_with_storage(
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        max_degree: u32,
        max_extra: usize,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        use rayon::prelude::*;
        let mut graph =
            Self::try_empty_with_storage(partitions.len(), max_degree, max_extra, storage)?;
        for (node, (local, remote, extra)) in partitions.iter().enumerate() {
            graph.validate_partition(node, local, remote, extra)?;
        }
        graph
            .buffer
            .par_chunks_mut(graph.stride)
            .zip(partitions.par_iter())
            .for_each(|(row, (local, remote, extra))| {
                Self::write_row(row, local, remote, extra);
            });
        Ok(graph)
    }

    pub(crate) fn build_from_partitions(
        partitions: &[(Vec<u32>, Vec<u32>, Vec<u32>)],
        max_degree: u32,
        max_extra: usize,
    ) -> Self {
        Self::try_from_partitions(partitions, max_degree, max_extra)
            .expect("invalid graph partitions")
    }
    #[inline]
    pub(crate) fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    pub(crate) fn stride(&self) -> usize {
        self.stride
    }

    /// Raw slab as bytes — used by benchmarks to `mlock` the graph
    /// pages so they cannot be swapped between warmup and the timed
    /// sweep.
    #[inline]
    pub(crate) fn buffer_bytes(&self) -> &[u8] {
        let n = self.buffer.len() * std::mem::size_of::<u32>();
        // SAFETY: all u32 words (including padding) are initialized. The byte view
        // spans exactly this allocation and cannot outlive its immutable borrow.
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr() as *const u8, n) }
    }

    #[inline]
    fn slot(&self, i: usize) -> &[u32] {
        let base = i * self.stride;
        &self.buffer[base..base + self.stride]
    }

    /// Total graph degree.
    #[inline]
    pub(crate) fn degree(&self, i: usize) -> usize {
        self.slot(i)[0] as usize
    }

    /// Number of extra candidates (not in graph).
    #[inline]
    pub(crate) fn extra_count(&self, i: usize) -> usize {
        self.slot(i)[1] as usize
    }

    /// Per-node local count (base t + promoted common).
    #[inline]
    pub(crate) fn local_count(&self, i: usize) -> usize {
        self.slot(i)[2] as usize
    }

    /// All graph neighbors = local + remote. Used for **navigation**.
    #[inline]
    pub(crate) fn neighbors(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let deg = s[0] as usize;
        &s[HEADER_U32..HEADER_U32 + deg]
    }

    /// Local neighbors (candidate-set-confirmed). Navigation + reranking.
    #[inline]
    pub(crate) fn local_neighbors(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let lc = s[2] as usize;
        &s[HEADER_U32..HEADER_U32 + lc]
    }

    /// Remote neighbors (long-range shortcuts). Navigation only.
    #[inline]
    pub(crate) fn remote_neighbors(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let deg = s[0] as usize;
        let lc = s[2] as usize;
        &s[HEADER_U32 + lc..HEADER_U32 + deg]
    }

    /// Extra candidates (in candidate set, not in graph). Reranking only.
    #[inline]
    pub(crate) fn extra_candidates(&self, i: usize) -> &[u32] {
        let s = self.slot(i);
        let deg = s[0] as usize;
        let extra = s[1] as usize;
        &s[HEADER_U32 + deg..HEADER_U32 + deg + extra]
    }

    /// Reranking slices: (local_neighbors, extra_candidates).
    #[inline]
    pub(crate) fn rerank_candidates(&self, i: usize) -> (&[u32], &[u32]) {
        (self.local_neighbors(i), self.extra_candidates(i))
    }

    /// Total reranking candidate count = local + extra.
    #[inline]
    pub(crate) fn rerank_count(&self, i: usize) -> usize {
        let s = self.slot(i);
        s[2] as usize + s[1] as usize
    }

    /// Check if edge (from → to) exists in the graph.
    #[inline]
    pub(crate) fn contains_edge(&self, from: u32, to: u32) -> bool {
        self.neighbors(from as usize).contains(&to)
    }

    /// Prefetch node slot into L1 cache.
    #[inline]
    pub(crate) fn prefetch_node(&self, i: usize) {
        orion_ann_vector::prefetch_vector(self.slot(i));
    }

    pub(crate) fn print_stats(&self) {
        let n = self.num_nodes;
        if n == 0 {
            return;
        }
        let total_deg: usize = (0..n).map(|i| self.degree(i)).sum();
        let total_extra: usize = (0..n).map(|i| self.extra_count(i)).sum();
        let total_local: usize = (0..n).map(|i| self.local_count(i)).sum();
        let total_rerank: usize = (0..n).map(|i| self.rerank_count(i)).sum();

        log::info!("PhasedGraph ({} nodes):", n);
        log::info!(
            "  R={}, base_local={}, stride={} u32",
            self.max_degree,
            self.base_local_count,
            self.stride
        );
        log::info!("  Avg degree:       {:.1}", total_deg as f64 / n as f64);
        log::info!(
            "  Avg local:        {:.1} (base + promoted)",
            total_local as f64 / n as f64
        );
        log::info!("  Avg extra:        {:.1}", total_extra as f64 / n as f64);
        log::info!(
            "  Avg rerank:       {:.1} (local+extra)",
            total_rerank as f64 / n as f64
        );
        log::info!(
            "  Memory:           {:.1} MB",
            (self.buffer.len() * 4) as f64 / 1_048_576.0,
        );
    }
}

impl Clone for PhasedGraph {
    fn clone(&self) -> Self {
        let mut copy = Self::allocate(
            self.num_nodes,
            self.max_degree,
            self.max_extra,
            self.stride,
            self.base_local_count,
        )
        .expect("graph allocation failed");
        copy.buffer
            .as_mut_slice()
            .copy_from_slice(self.buffer.as_slice());
        copy
    }
}
