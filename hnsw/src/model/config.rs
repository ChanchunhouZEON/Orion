/// HNSW configuration parameters.
#[derive(Debug, Clone)]
pub struct HNSWConfig {
    /// Max number of connections per element per layer (except layer 0).
    pub m: usize,
    /// Max number of connections per element at layer 0.
    pub m_max0: usize,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search.
    pub ef_search: usize,
    /// Level multiplier: ml = 1 / ln(M).
    pub ml: f64,
    /// Number of threads for parallel construction.
    pub num_threads: u32,
}

impl HNSWConfig {
    pub fn new(m: usize, ef_construction: usize, ef_search: usize, num_threads: u32) -> Self {
        Self {
            m,
            m_max0: m * 2,
            ef_construction,
            ef_search,
            ml: 1.0 / (m as f64).ln(),
            num_threads,
        }
    }
}

impl Default for HNSWConfig {
    fn default() -> Self {
        Self::new(16, 200, 100, 1)
    }
}
