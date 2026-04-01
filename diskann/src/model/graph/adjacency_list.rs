/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::ops::{Deref, DerefMut};

#[derive(Debug, Eq, PartialEq)]
pub struct AdjacencyList {
    edges: Vec<u32>,
}

const GRAPH_SLACK_FACTOR: f32 = 1.3_f32;

impl AdjacencyList {
    pub fn for_range(range: usize) -> Self {
        let capacity = (range as f32 * GRAPH_SLACK_FACTOR).ceil() as usize;
        Self {
            edges: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, node_id: u32) {
        debug_assert!(self.edges.len() < self.edges.capacity());
        self.edges.push(node_id);
    }

    /// Consume the adjacency list and return the underlying `Vec<u32>`.
    #[inline(always)]
    pub fn into_vec(self) -> Vec<u32> {
        self.edges
    }
}

impl From<Vec<u32>> for AdjacencyList {
    fn from(edges: Vec<u32>) -> Self {
        Self { edges }
    }
}

impl Clone for AdjacencyList {
    fn clone(&self) -> Self {
        Self {
            edges: self.edges.clone(),
        }
    }
}

impl Deref for AdjacencyList {
    type Target = Vec<u32>;

    fn deref(&self) -> &Self::Target {
        &self.edges
    }
}

impl DerefMut for AdjacencyList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.edges
    }
}

impl<'a> IntoIterator for &'a AdjacencyList {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.edges.iter()
    }
}
