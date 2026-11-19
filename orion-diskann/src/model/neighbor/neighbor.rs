/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::cmp::Ordering;

/// One candidate ID, ascending distance and traversal state.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    /// Internal graph ID associated with this candidate.
    pub id: u32,

    /// Smaller-is-closer distance used for candidate ordering.
    pub distance: f32,

    /// Whether this candidate has already been expanded.
    pub visited: bool,
}

impl Neighbor {
    /// Creates an unvisited scored candidate for insertion into the beam.
    pub fn new(id: u32, distance: f32) -> Self {
        Self {
            id,
            distance,
            visited: false,
        }
    }
}

impl Default for Neighbor {
    /// Creates an unvisited zero-valued placeholder for reusable candidate buffers.
    fn default() -> Self {
        Self {
            id: 0,
            distance: 0.0_f32,
            visited: false,
        }
    }
}

impl PartialEq for Neighbor {
    /// Compares candidate identity by internal ID, independently of distance and visit state.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    /// Orders candidates by distance and then internal ID for deterministic beam ties.
    fn cmp(&self, other: &Self) -> Ordering {
        let ord = self
            .distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal);
        if ord == Ordering::Equal {
            return self.id.cmp(&other.id);
        }
        ord
    }
}

#[expect(
    clippy::non_canonical_partial_ord_impl,
    reason = "Preserve the legacy eq/lt-only comparison contract and its explicit partial_cmp rejection."
)]
impl PartialOrd for Neighbor {
    /// Orders candidates by distance and then internal ID for deterministic beam ties.
    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.distance < other.distance || (self.distance == other.distance && self.id < other.id)
    }

    /// Rejects general partial comparison; this legacy type supports only its explicit equality and less-than operations.
    #[allow(clippy::panic)]
    fn partial_cmp(&self, _: &Self) -> Option<Ordering> {
        panic!("Neighbor only allows eq and lt")
    }
}

#[cfg(test)]
mod neighbor_test {
    use super::*;

    #[test]
    fn eq_lt_works() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);
        let n3 = Neighbor::new(1, 1.1);
        assert!(n1 != n2);
        assert!(n1 < n2);
        assert!(n1 == n3);
    }

    #[test]
    #[should_panic]
    fn gt_should_panic() {
        let n1 = Neighbor::new(1, 1.1);
        let n2 = Neighbor::new(2, 2.0);
        assert!(n2 > n1);
    }
}
