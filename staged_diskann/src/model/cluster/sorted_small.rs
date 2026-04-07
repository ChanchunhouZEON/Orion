/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

// ─── SortedSmallSet ──────────────────────────────────────────────────────────

/// Sorted `Vec<u32>` set. Zero allocation when empty; grows on demand.
///
/// Uses binary search for lookups and linear shift for insert/remove.
/// Optimal for small collections (≤16 elements) where hashing overhead
/// dominates.
#[derive(Clone)]
pub struct SortedSmallSet {
    data: Vec<u32>,
}

impl SortedSmallSet {
    #[inline]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Create with a single initial element.
    #[inline]
    pub fn with_one(val: u32) -> Self {
        Self { data: vec![val] }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn contains(&self, val: u32) -> bool {
        self.data.binary_search(&val).is_ok()
    }

    /// Insert `val` in sorted order. Returns true if newly inserted.
    #[inline]
    pub fn insert(&mut self, val: u32) -> bool {
        match self.data.binary_search(&val) {
            Ok(_) => false,
            Err(pos) => {
                self.data.insert(pos, val);
                true
            }
        }
    }

    /// Remove `val`. Returns true if it was present.
    #[inline]
    pub fn remove(&mut self, val: u32) -> bool {
        match self.data.binary_search(&val) {
            Ok(pos) => {
                self.data.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    /// Extend with all elements from another set (union).
    pub fn extend_from(&mut self, other: &SortedSmallSet) {
        for &val in &other.data {
            self.insert(val);
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    #[inline]
    pub fn as_slice(&self) -> &[u32] {
        &self.data
    }

    pub fn iter(&self) -> std::slice::Iter<'_, u32> {
        self.data.iter()
    }

    /// Create from an iterator of u32 values.
    pub fn from_iter(iter: impl IntoIterator<Item = u32>) -> Self {
        let mut s = Self::new();
        for val in iter {
            s.insert(val);
        }
        s
    }
}

impl std::fmt::Debug for SortedSmallSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.data.iter()).finish()
    }
}

// ─── SortedSmallMap ──────────────────────────────────────────────────────────

/// Sorted `Vec`-backed map from `u32` keys to `V` values.
/// Zero allocation when empty; keys and values grow together on demand.
pub struct SortedSmallMap<V> {
    keys: Vec<u32>,
    values: Vec<V>,
}

impl<V> SortedSmallMap<V> {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    #[inline]
    pub fn contains_key(&self, key: u32) -> bool {
        self.keys.binary_search(&key).is_ok()
    }

    pub fn get(&self, key: u32) -> Option<&V> {
        match self.keys.binary_search(&key) {
            Ok(pos) => Some(&self.values[pos]),
            Err(_) => None,
        }
    }

    pub fn get_mut(&mut self, key: u32) -> Option<&mut V> {
        match self.keys.binary_search(&key) {
            Ok(pos) => Some(&mut self.values[pos]),
            Err(_) => None,
        }
    }

    /// Insert key-value pair. Returns `None` if newly inserted,
    /// `Some(old_value)` if key already existed (value is replaced).
    pub fn insert(&mut self, key: u32, value: V) -> Option<V> {
        match self.keys.binary_search(&key) {
            Ok(pos) => Some(std::mem::replace(&mut self.values[pos], value)),
            Err(pos) => {
                self.keys.insert(pos, key);
                self.values.insert(pos, value);
                None
            }
        }
    }

    /// Remove entry by key. Returns the value if it existed.
    pub fn remove(&mut self, key: u32) -> Option<V> {
        match self.keys.binary_search(&key) {
            Ok(pos) => {
                self.keys.remove(pos);
                Some(self.values.remove(pos))
            }
            Err(_) => None,
        }
    }

    #[inline]
    pub fn key_slice(&self) -> &[u32] {
        &self.keys
    }

    #[inline]
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }

    /// Iterate over (key, &value) pairs in sorted key order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &V)> {
        self.keys.iter().copied().zip(self.values.iter())
    }

    /// Drain all entries, returning an iterator of (key, value) pairs.
    pub fn drain(&mut self) -> impl Iterator<Item = (u32, V)> + '_ {
        self.keys.drain(..).zip(self.values.drain(..))
    }

    pub fn max_by_key<F, K: Ord>(&self, mut f: F) -> Option<(u32, &V)>
    where
        F: FnMut(&V) -> K,
    {
        self.keys
            .iter()
            .zip(self.values.iter())
            .max_by_key(|(_, v)| f(v))
            .map(|(&k, v)| (k, v))
    }

    pub fn min_by_key<F, K: Ord>(&self, mut f: F) -> Option<(u32, &V)>
    where
        F: FnMut(&V) -> K,
    {
        self.keys
            .iter()
            .zip(self.values.iter())
            .min_by_key(|(_, v)| f(v))
            .map(|(&k, v)| (k, v))
    }
}

impl<V: std::fmt::Debug> std::fmt::Debug for SortedSmallMap<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(self.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_basic() {
        let mut s = SortedSmallSet::new();
        assert!(s.insert(5));
        assert!(s.insert(2));
        assert!(s.insert(8));
        assert!(!s.insert(5)); // duplicate
        assert_eq!(s.len(), 3);
        assert_eq!(s.as_slice(), &[2, 5, 8]);
        assert!(s.contains(5));
        assert!(!s.contains(3));
    }

    #[test]
    fn test_set_remove() {
        let mut s = SortedSmallSet::from_iter([1, 3, 5, 7]);
        assert!(s.remove(3));
        assert_eq!(s.as_slice(), &[1, 5, 7]);
        assert!(!s.remove(3)); // already removed
        assert!(s.remove(1));
        assert_eq!(s.as_slice(), &[5, 7]);
    }

    #[test]
    fn test_set_extend() {
        let mut a = SortedSmallSet::from_iter([1, 3]);
        let b = SortedSmallSet::from_iter([2, 3, 4]);
        a.extend_from(&b);
        assert_eq!(a.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_map_basic() {
        let mut m: SortedSmallMap<i32> = SortedSmallMap::new();
        assert!(m.insert(5, 50).is_none());
        assert!(m.insert(2, 20).is_none());
        assert!(m.insert(8, 80).is_none());
        assert_eq!(m.len(), 3);
        assert_eq!(*m.get(5).unwrap(), 50);
        assert_eq!(*m.get(2).unwrap(), 20);
        assert!(m.get(3).is_none());
        assert_eq!(m.key_slice(), &[2, 5, 8]);
    }

    #[test]
    fn test_map_remove() {
        let mut m: SortedSmallMap<i32> = SortedSmallMap::new();
        m.insert(1, 10);
        m.insert(3, 30);
        m.insert(5, 50);
        assert_eq!(m.remove(3), Some(30));
        assert_eq!(m.len(), 2);
        assert_eq!(m.key_slice(), &[1, 5]);
    }

    #[test]
    fn test_map_replace() {
        let mut m: SortedSmallMap<i32> = SortedSmallMap::new();
        m.insert(1, 10);
        let old = m.insert(1, 100);
        assert_eq!(old, Some(10));
        assert_eq!(*m.get(1).unwrap(), 100);
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_map_iter() {
        let mut m: SortedSmallMap<i32> = SortedSmallMap::new();
        m.insert(3, 30);
        m.insert(1, 10);
        m.insert(2, 20);
        let pairs: Vec<_> = m.iter().map(|(k, &v)| (k, v)).collect();
        assert_eq!(pairs, vec![(1, 10), (2, 20), (3, 30)]);
    }
}
