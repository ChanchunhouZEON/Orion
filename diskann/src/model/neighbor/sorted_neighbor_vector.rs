/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::Neighbor;
use std::ops::{Deref, DerefMut};

#[derive(Debug)]
pub struct SortedNeighborVector<'a>(&'a mut Vec<Neighbor>);

impl<'a> SortedNeighborVector<'a> {
    pub fn new(vec: &'a mut Vec<Neighbor>) -> Self {
        vec.sort_unstable();
        Self(vec)
    }
}

impl<'a> Deref for SortedNeighborVector<'a> {
    type Target = Vec<Neighbor>;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<'a> DerefMut for SortedNeighborVector<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}
