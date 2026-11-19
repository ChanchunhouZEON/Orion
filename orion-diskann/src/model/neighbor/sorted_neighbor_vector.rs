/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::Neighbor;
use std::ops::{Deref, DerefMut};

/// Borrowed candidate vector adapter used by the sorted pruning path.
#[derive(Debug)]
pub struct SortedNeighborVector<'a>(&'a mut Vec<Neighbor>);

impl<'a> SortedNeighborVector<'a> {
    /// Sorts the borrowed candidate vector in place before exposing pruning operations.
    pub fn new(vec: &'a mut Vec<Neighbor>) -> Self {
        vec.sort_unstable();
        Self(vec)
    }
}

impl<'a> Deref for SortedNeighborVector<'a> {
    type Target = Vec<Neighbor>;

    /// Borrows the borrowed candidate vector without copying it.
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> DerefMut for SortedNeighborVector<'a> {
    /// Exclusively borrows the borrowed candidate vector for in-place mutation.
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}
