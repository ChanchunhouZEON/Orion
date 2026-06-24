/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::cmp::Ordering;

/// Neighbor node with distance and visited state.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
    pub visited: bool,
}

impl Neighbor {
    pub fn new(id: u32, distance: f32) -> Self {
        Self {
            id,
            distance,
            visited: false,
        }
    }
}

impl Default for Neighbor {
    fn default() -> Self {
        Self {
            id: 0,
            distance: 0.0,
            visited: false,
        }
    }
}

impl PartialEq for Neighbor {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Neighbor {}

impl Ord for Neighbor {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        let ord = self.distance.total_cmp(&other.distance);
        if ord == Ordering::Equal {
            self.id.cmp(&other.id)
        } else {
            ord
        }
    }
}

impl PartialOrd for Neighbor {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordering_by_distance() {
        let a = Neighbor::new(0, 1.0);
        let b = Neighbor::new(1, 2.0);
        assert!(a < b);
        assert!(!(b < a));
    }

    #[test]
    fn test_ordering_tie_break_by_id() {
        let a = Neighbor::new(0, 1.0);
        let b = Neighbor::new(1, 1.0);
        assert!(a < b);
    }

    #[test]
    fn test_equality_by_id() {
        let a = Neighbor::new(5, 1.0);
        let b = Neighbor::new(5, 99.0);
        assert_eq!(a, b); // equality is by id only
    }

    #[test]
    fn test_default() {
        let n = Neighbor::default();
        assert_eq!(n.id, 0);
        assert_eq!(n.distance, 0.0);
        assert!(!n.visited);
    }

    #[test]
    fn test_new_not_visited() {
        let n = Neighbor::new(1, 2.0);
        assert!(!n.visited);
    }

    #[test]
    fn test_partial_cmp_does_not_panic() {
        let a = Neighbor::new(0, 1.0);
        let b = Neighbor::new(1, 2.0);
        assert_eq!(a.partial_cmp(&b), Some(Ordering::Less));
        assert_eq!(b.partial_cmp(&a), Some(Ordering::Greater));
    }

    #[test]
    fn test_sort_neighbors() {
        let mut v = vec![
            Neighbor::new(3, 5.0),
            Neighbor::new(1, 1.0),
            Neighbor::new(2, 3.0),
            Neighbor::new(0, 1.0),
        ];
        v.sort();
        assert_eq!(v[0].id, 0);
        assert_eq!(v[1].id, 1);
        assert_eq!(v[2].id, 2);
        assert_eq!(v[3].id, 3);
    }
}
