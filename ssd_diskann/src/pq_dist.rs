use diskann::model::FixedChunkPQTable;
use std::sync::Arc;

/// Lightweight PQ distance computer that stores only codes + codebooks in memory.
/// For SIFT-1M with 8 chunks: ~8MB codes + ~131KB tables = ~8.1MB total.
pub struct PQDistanceComputer {
    pq: Arc<FixedChunkPQTable>,
    /// Flattened PQ codes: [num_points * num_chunks] stored as Vec<u8> for compact memory.
    codes: Vec<u8>,
    num_points: usize,
    num_chunks: usize,
}

impl PQDistanceComputer {
    /// Create from a trained FixedChunkPQTable and externally stored codes.
    pub fn new(pq: Arc<FixedChunkPQTable>, codes: Vec<u8>) -> Self {
        let num_chunks = pq.get_num_chunks();
        let num_points = if num_chunks > 0 {
            codes.len() / num_chunks
        } else {
            0
        };

        Self {
            pq,
            codes,
            num_points,
            num_chunks,
        }
    }

    /// Preprocess query and compute per-chunk distance tables.
    /// Returns the flat distance table for use with `adc_distance`.
    pub fn compute_adc_table(&self, query: &[f32]) -> Vec<f32> {
        let mut query_vec = query.to_vec();
        self.pq.preprocess_query(&mut query_vec);
        self.pq.populate_chunk_distances(&query_vec)
    }

    /// Compute approximate distance for a point using precomputed ADC tables.
    pub fn adc_distance(&self, point_id: u32, pq_dists: &[f32]) -> f32 {
        let pid = point_id as usize;
        if pid >= self.num_points {
            return f32::MAX;
        }
        let offset = pid * self.num_chunks;
        let code = &self.codes[offset..offset + self.num_chunks];
        self.pq.adc_distance(code, pq_dists)
    }

    /// Number of points with PQ codes.
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// Approximate memory usage in bytes (codes + PQ table).
    pub fn memory_bytes(&self) -> usize {
        // codes: num_points * num_chunks bytes
        let codes_bytes = self.codes.len();
        // PQ table: 256 * dim * 4 bytes + centroids: dim * 4 bytes
        let dim = self.pq.dim();
        let table_bytes = 256 * dim * 4 + dim * 4;
        codes_bytes + table_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_pq_distance_computer() {
        let n = 500;
        let dim = 16;
        let m = 4;
        let data =
            Array2::from_shape_fn((n, dim), |(i, j)| (i * dim + j) as f32 / (n * dim) as f32);
        let flat: Vec<f32> = data.as_slice().unwrap().to_vec();

        let pq = FixedChunkPQTable::train(&flat, n, dim, m);
        let codes = pq.encode(&flat, n);

        let computer = PQDistanceComputer::new(Arc::new(pq), codes);
        assert_eq!(computer.num_points(), n);

        // Compute ADC for first row
        let query: Vec<f32> = data.row(0).to_vec();
        let tables = computer.compute_adc_table(&query);
        let self_dist = computer.adc_distance(0, &tables);
        // Self-distance should be small (quantization error)
        assert!(
            self_dist < 1.0,
            "Self-distance {} should be small",
            self_dist
        );

        // Memory should be reasonable
        let mem = computer.memory_bytes();
        assert!(mem > 0);
        assert!(mem < 1_000_000); // Well under 1MB for 500 points
    }
}
