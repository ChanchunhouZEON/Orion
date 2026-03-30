use crate::disk_layout;
use crate::disk_reader::DiskReader;
use crate::pq_dist::PQDistanceComputer;
use diskann::model::{FixedChunkPQTable, InMemoryGraph};
use hashbrown::HashSet;
use ndarray::ArcArray2;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use vector::{FullPrecisionDistance, Metric};

/// A candidate node with distance, ordered by distance (min-heap via Reverse).
#[derive(Clone)]
struct Candidate {
    id: u32,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap (smallest distance first)
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// SSD-based DiskANN index.
/// Keeps only PQ codes in memory; full vectors and graph are read from disk.
pub struct SSDIndex<const N: usize>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    disk_reader: DiskReader,
    pq_dist: PQDistanceComputer,
    medoid: u32,
    beam_width: usize,
    search_list_size: usize,
}

impl<const N: usize> SSDIndex<N>
where
    [f32; N]: FullPrecisionDistance<f32, N>,
{
    /// Build an SSD index from in-memory data and graph.
    ///
    /// 1. Writes the disk index file
    /// 2. Opens it for mmap reading
    /// 3. Creates PQ distance computer from the trained PQ
    /// 4. Drops all in-memory data (caller should drop data/graph after this)
    pub fn build(
        data: &ArcArray2<f32>,
        graph: &InMemoryGraph,
        medoid: u32,
        pq: Arc<FixedChunkPQTable>,
        pq_codes: Vec<u8>,
        disk_path: &Path,
        beam_width: usize,
        search_list_size: usize,
    ) -> anyhow::Result<Self> {
        assert_eq!(
            data.ncols(),
            N,
            "Data dimension {} != const N={}",
            data.ncols(),
            N
        );

        // Write disk index
        log::info!("Writing SSD disk index to {:?}...", disk_path);
        let _meta = disk_layout::write_disk_index(disk_path, data, graph, medoid)?;

        // Open for reading
        let disk_reader = DiskReader::open(disk_path)?;
        let pq_dist = PQDistanceComputer::new(pq, pq_codes);

        log::info!(
            "SSD index ready: {} points, dim={}, PQ memory={:.2}MB",
            disk_reader.num_points(),
            disk_reader.dimension(),
            pq_dist.memory_bytes() as f64 / (1024.0 * 1024.0)
        );

        Ok(Self {
            disk_reader,
            pq_dist,
            medoid,
            beam_width,
            search_list_size,
        })
    }

    /// Open an existing SSD index from disk.
    pub fn open(
        disk_path: &Path,
        pq: Arc<FixedChunkPQTable>,
        pq_codes: Vec<u8>,
        beam_width: usize,
        search_list_size: usize,
    ) -> anyhow::Result<Self> {
        let disk_reader = DiskReader::open(disk_path)?;
        let medoid = disk_reader.medoid();
        let pq_dist = PQDistanceComputer::new(pq, pq_codes);

        Ok(Self {
            disk_reader,
            pq_dist,
            medoid,
            beam_width,
            search_list_size,
        })
    }

    /// PQ-guided beam search with disk reads for exact reranking.
    ///
    /// Algorithm:
    /// 1. Compute ADC tables for the query
    /// 2. Start from medoid, use PQ distances to expand candidates
    /// 3. For top beam_width candidates, read exact vectors + neighbors from disk
    /// 4. Compute exact L2 distances, update best candidates
    /// 5. Add unvisited neighbors to PQ candidate queue
    /// 6. Return top-k by exact distance
    pub fn search(&self, query: &[f32; N], k: usize) -> Vec<u32> {
        let pq_dists = self.pq_dist.compute_adc_table(query.as_slice());

        let mut visited = HashSet::with_capacity(self.search_list_size * 2);
        let mut pq_candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut best_exact: Vec<(u32, f32)> = Vec::with_capacity(self.search_list_size);

        // Seed with medoid
        let medoid_dist = self.pq_dist.adc_distance(self.medoid, &pq_dists);
        pq_candidates.push(Candidate {
            id: self.medoid,
            distance: medoid_dist,
        });
        visited.insert(self.medoid);

        let mut iterations = 0;
        let max_iterations = self.search_list_size * 4;

        while !pq_candidates.is_empty() && iterations < max_iterations {
            // Collect up to beam_width candidates to read from disk
            let mut beam: Vec<Candidate> = Vec::with_capacity(self.beam_width);
            while beam.len() < self.beam_width {
                if let Some(c) = pq_candidates.pop() {
                    beam.push(c);
                } else {
                    break;
                }
            }

            if beam.is_empty() {
                break;
            }

            for candidate in &beam {
                iterations += 1;

                // Read exact vector + neighbors from disk
                if let Some(node) = self.disk_reader.read_node(candidate.id) {
                    // Compute exact L2 distance
                    let vec: [f32; N] = DiskReader::parse_vector_array(node.vector_bytes);
                    let exact_dist = <[f32; N]>::distance_compare(query, &vec, Metric::L2);

                    best_exact.push((candidate.id, exact_dist));

                    // Add unvisited neighbors to PQ candidate queue
                    for &nbr in &node.neighbors {
                        if nbr < self.pq_dist.num_points() as u32 && visited.insert(nbr) {
                            let nbr_dist = self.pq_dist.adc_distance(nbr, &pq_dists);
                            pq_candidates.push(Candidate {
                                id: nbr,
                                distance: nbr_dist,
                            });
                        }
                    }
                }
            }

            // Early termination: if we have enough exact results and the best PQ candidate
            // is worse than our k-th best exact result, we can stop
            if best_exact.len() >= k {
                best_exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
                if let Some(top_pq) = pq_candidates.peek() {
                    let kth_exact = best_exact[k - 1].1;
                    // PQ distance is a lower bound; if best PQ candidate is worse, stop
                    if top_pq.distance > kth_exact * 1.5 && iterations >= self.search_list_size {
                        break;
                    }
                }
            }
        }

        // Sort by exact distance and return top-k
        best_exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        best_exact.iter().take(k).map(|(id, _)| *id).collect()
    }

    /// Memory usage (heap) — only PQ codes + tables.
    pub fn memory_bytes(&self) -> usize {
        self.pq_dist.memory_bytes()
    }

    /// Path to the disk index file.
    pub fn disk_path(&self) -> PathBuf {
        // We don't store the path; callers track it
        PathBuf::new()
    }

    /// Warm the OS page cache by reading the medoid and its neighbors.
    pub fn warm_cache(&self, max_hops: usize) {
        let mut queue = vec![self.medoid];
        let mut visited = HashSet::new();
        visited.insert(self.medoid);

        for _ in 0..max_hops {
            let mut next_queue = Vec::new();
            for &node_id in &queue {
                self.disk_reader.prefetch_node(node_id);
                if let Some(node) = self.disk_reader.read_node(node_id) {
                    for &nbr in &node.neighbors {
                        if visited.insert(nbr) {
                            next_queue.push(nbr);
                        }
                    }
                }
            }
            queue = next_queue;
        }
        log::info!("Warmed cache: {} nodes prefetched", visited.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn build_test_index() -> (SSDIndex<16>, Vec<[f32; 16]>) {
        let n = 500;
        let dim = 16;
        let m = 4;
        let degree = 8;

        // Generate random-ish data
        let data = Array2::from_shape_fn((n, dim), |(i, j)| {
            ((i * 17 + j * 31) % 1000) as f32 / 1000.0
        });
        let arc_data = data.into_shared();

        // Build a simple graph (each node connects to next `degree` nodes)
        use diskann::model::graph::AdjacencyList;
        let graph = InMemoryGraph::new(n, degree as u32);
        for i in 0..n {
            let mut nbrs = Vec::with_capacity(degree);
            for d in 1..=degree {
                nbrs.push(((i + d) % n) as u32);
            }
            let mut vertex = graph
                .write_vertex_and_neighbors(i as u32)
                .expect("write lock");
            vertex.set_neighbors(AdjacencyList::from(nbrs));
        }

        // Train PQ
        let flat: Vec<f32> = arc_data.as_slice().unwrap().to_vec();
        let pq = FixedChunkPQTable::train(&flat, n, dim, m);
        let pq_codes = pq.encode(&flat, n);
        let pq = Arc::new(pq);

        // Write to temp file
        let dir = std::env::temp_dir();
        let path = dir.join("test_ssd_index.bin");

        let index = SSDIndex::<16>::build(&arc_data, &graph, 0, pq, pq_codes, &path, 4, 64)
            .expect("build SSD index");

        // Prepare query arrays
        let queries: Vec<[f32; 16]> = (0..10)
            .map(|i| {
                let mut q = [0.0f32; 16];
                let row = arc_data.row(i);
                q.copy_from_slice(row.as_slice().unwrap());
                q
            })
            .collect();

        (index, queries)
    }

    #[test]
    fn test_ssd_search_returns_results() {
        let (index, queries) = build_test_index();

        for query in &queries {
            let results = index.search(query, 10);
            assert!(!results.is_empty(), "Search should return results");
            assert!(results.len() <= 10);
        }
    }

    #[test]
    fn test_ssd_self_search() {
        let (index, queries) = build_test_index();

        // Searching for the first point should find itself (or very close)
        let results = index.search(&queries[0], 1);
        assert!(!results.is_empty());
        // The result should contain point 0 (exact match)
        assert_eq!(results[0], 0, "Self-search should find the query point");
    }

    #[test]
    fn test_ssd_memory_bytes() {
        let (index, _) = build_test_index();
        let mem = index.memory_bytes();
        // 500 points * 4 chunks = 2000 bytes codes + codebook overhead
        assert!(mem > 2000);
        assert!(mem < 1_000_000);
    }
}
