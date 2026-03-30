/// NSG configuration parameters.
#[derive(Debug, Clone)]
pub struct NSGConfig {
    /// Maximum out-degree R.
    pub r: usize,
    /// Search list size L for construction and search.
    pub l: usize,
    /// Candidate pool size C for edge selection.
    pub c: usize,
    /// Number of neighbors in initial k-NN graph.
    pub k: usize,
    /// Number of threads for parallel construction.
    pub num_threads: u32,
}

impl NSGConfig {
    pub fn new(r: usize, l: usize, c: usize, k: usize, num_threads: u32) -> Self {
        Self {
            r,
            l,
            c,
            k,
            num_threads,
        }
    }
}

impl Default for NSGConfig {
    fn default() -> Self {
        Self::new(32, 100, 200, 50, 1)
    }
}
