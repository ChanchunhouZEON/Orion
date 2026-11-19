/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::ops::{Deref, DerefMut};

/// Owned neighbor IDs for one mutable graph vertex.
#[derive(Debug, Eq, PartialEq)]
pub struct AdjacencyList {
    edges: Vec<u32>,
}

const GRAPH_SLACK_FACTOR: f32 = 1.3_f32;

impl AdjacencyList {
    /// Creates an empty neighbor-ID list with the requested reserved capacity.
    pub fn for_range(range: usize) -> Self {
        let capacity = (range as f32 * GRAPH_SLACK_FACTOR).ceil() as usize;
        Self {
            edges: Vec::with_capacity(capacity),
        }
    }

    /// Appends one neighbor ID without performing duplicate or graph-bound validation.
    pub fn push(&mut self, node_id: u32) {
        debug_assert!(self.edges.len() < self.edges.capacity());
        self.edges.push(node_id);
    }

    /// Insert at position, shifting later elements right.
    pub fn insert(&mut self, index: usize, node_id: u32) {
        self.edges.insert(index, node_id);
    }

    /// Consume the adjacency list and return the underlying `Vec<u32>`.
    #[inline(always)]
    pub fn into_vec(self) -> Vec<u32> {
        self.edges
    }
}

impl From<Vec<u32>> for AdjacencyList {
    /// Takes ownership of the supplied edge vector without sorting or copying it.
    fn from(edges: Vec<u32>) -> Self {
        Self { edges }
    }
}

impl Clone for AdjacencyList {
    /// Copies neighbor IDs into an independent adjacency list.
    fn clone(&self) -> Self {
        Self {
            edges: self.edges.clone(),
        }
    }
}

impl Deref for AdjacencyList {
    type Target = Vec<u32>;

    /// Borrows the owned edge vector without copying it.
    fn deref(&self) -> &Self::Target {
        &self.edges
    }
}

impl DerefMut for AdjacencyList {
    /// Exclusively borrows the owned edge vector for in-place mutation.
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.edges
    }
}

impl<'a> IntoIterator for &'a AdjacencyList {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    /// Iterates borrowed neighbor IDs without copying the adjacency list.
    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}
