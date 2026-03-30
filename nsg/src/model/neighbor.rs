use std::cmp::Ordering;

/// A neighbor with an ID and distance.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub id: u32,
    pub distance: f32,
}

impl Neighbor {
    pub fn new(id: u32, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl PartialEq for Neighbor {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.distance == other.distance
    }
}

impl Eq for Neighbor {}

impl PartialOrd for Neighbor {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Neighbor {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ordering() {
        let a = Neighbor::new(0, 1.0);
        let b = Neighbor::new(1, 2.0);
        assert!(a < b);
    }

    #[test]
    fn test_equality() {
        let a = Neighbor::new(5, 3.14);
        let b = Neighbor::new(5, 3.14);
        assert_eq!(a, b);
    }

    #[test]
    fn test_sort() {
        let mut v = vec![
            Neighbor::new(2, 3.0),
            Neighbor::new(0, 1.0),
            Neighbor::new(1, 2.0),
        ];
        v.sort();
        assert_eq!(v[0].id, 0);
        assert_eq!(v[1].id, 1);
        assert_eq!(v[2].id, 2);
    }
}
