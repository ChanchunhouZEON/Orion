/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::storage::sector_layout::SectorLayout;
use memmap2::{Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use diskann::common::ANNResult;

/// Memory-mapped storage for zero-copy access to per-node graph data.
///
/// Per-node sector layout (redundant cluster page):
///   [neighbor_page] [data_page] [cluster_page]
///
/// The cluster_page is the same for all nodes in the same cluster,
/// stored redundantly so a single sequential SSD read gives everything
/// needed for both Phase 1 (full graph) and Phase 2 (compressed graph).
pub struct MmapStorage {
    mmap: Option<Mmap>,
    layout: SectorLayout,
}

impl std::fmt::Debug for MmapStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MmapStorage")
            .field("is_loaded", &self.mmap.is_some())
            .field("layout", &self.layout)
            .finish()
    }
}

/// Data for building a single cluster's page bytes.
/// Each record: [point_id: u32][neighbor_0: u32]...[neighbor_{max-1}: u32]
/// Neighbors are padded with 0 if out-degree < compressed_max_degree.
pub struct ClusterPageData {
    pub cluster_id: u32,
    /// (point_id, compressed_neighbors) for each point in the cluster.
    pub records: Vec<(u32, Vec<u32>)>,
}

impl MmapStorage {
    /// Create a new empty storage (for building, before write).
    pub fn new(layout: SectorLayout) -> Self {
        Self { mmap: None, layout }
    }

    /// Open an existing storage file for read-only access.
    pub fn open<P: AsRef<Path>>(path: P, layout: SectorLayout) -> ANNResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        Ok(Self {
            mmap: Some(mmap),
            layout,
        })
    }

    // ─── Per-node reads ───

    /// Read the neighbor page for a node (Phase 1).
    pub fn read_neighbor_page(&self, node_id: u32) -> Option<&[u8]> {
        let mmap = self.mmap.as_ref()?;
        let offset = self.layout.node_offset(node_id) as usize;
        let size = self.layout.neighbor_page_size();
        if offset + size <= mmap.len() {
            Some(&mmap[offset..offset + size])
        } else {
            None
        }
    }

    /// Read the data page for a node (both phases).
    pub fn read_data_page(&self, node_id: u32) -> Option<&[u8]> {
        let mmap = self.mmap.as_ref()?;
        let offset = self.layout.node_offset(node_id) as usize + self.layout.neighbor_page_size();
        let size = self.layout.data_page_size();
        if offset + size <= mmap.len() {
            Some(&mmap[offset..offset + size])
        } else {
            None
        }
    }

    /// Read neighbor_page + data_page together for a node (Phase 1 combined read).
    ///
    /// Since neighbor_page and data_page are adjacent in the sector layout,
    /// reading them as one contiguous region triggers a single page fault / SSD read.
    /// Returns (neighbor_page, data_page) slices.
    pub fn read_neighbor_and_data_pages(&self, node_id: u32) -> Option<(&[u8], &[u8])> {
        let mmap = self.mmap.as_ref()?;
        let base = self.layout.node_offset(node_id) as usize;
        let nbr_size = self.layout.neighbor_page_size();
        let data_size = self.layout.data_page_size();
        let total = nbr_size + data_size;
        if base + total <= mmap.len() {
            let neighbor_page = &mmap[base..base + nbr_size];
            let data_page = &mmap[base + nbr_size..base + total];
            Some((neighbor_page, data_page))
        } else {
            None
        }
    }

    /// Read the cluster neighbor page for a node (Phase 2).
    /// This page contains ALL points' compressed neighbors for the node's cluster.
    pub fn read_cluster_neighbor_page(&self, node_id: u32) -> Option<&[u8]> {
        let mmap = self.mmap.as_ref()?;
        let offset = self.layout.node_offset(node_id) as usize
            + self.layout.neighbor_page_size()
            + self.layout.data_page_size();
        let size = self.layout.cluster_page_size();
        if offset + size <= mmap.len() {
            Some(&mmap[offset..offset + size])
        } else {
            None
        }
    }

    /// Read data_page + cluster_page together for a node (Phase 2 combined read).
    ///
    /// Since data_page and cluster_page are adjacent in the sector layout,
    /// reading them as one contiguous region triggers a single page fault / SSD read,
    /// improving memory access efficiency during converged search.
    ///
    /// Returns (data_page, cluster_page) slices.
    pub fn read_data_and_cluster_pages(&self, node_id: u32) -> Option<(&[u8], &[u8])> {
        let mmap = self.mmap.as_ref()?;
        let base = self.layout.node_offset(node_id) as usize + self.layout.neighbor_page_size();
        let data_size = self.layout.data_page_size();
        let cluster_size = self.layout.cluster_page_size();
        let total = data_size + cluster_size;
        if base + total <= mmap.len() {
            let data_page = &mmap[base..base + data_size];
            let cluster_page = &mmap[base + data_size..base + total];
            Some((data_page, cluster_page))
        } else {
            None
        }
    }

    // ─── Cluster page parsing (static) ───

    /// Parse a specific point's compressed neighbors from a cluster page.
    ///
    /// Cluster page format:
    ///   [num_records: u32]
    ///   [record 0: point_id(u32) | neighbor_0(u32) | ... | neighbor_{max-1}(u32)]
    ///   ...
    ///
    /// Returns the neighbor IDs (excluding zero-padding) for the given point.
    pub fn parse_point_neighbors(
        page: &[u8],
        point_id: u32,
        compressed_max_degree: u32,
    ) -> Vec<u32> {
        if page.len() < 4 {
            return Vec::new();
        }
        let num_records = u32::from_le_bytes(page[0..4].try_into().unwrap()) as usize;
        let record_size = (1 + compressed_max_degree as usize) * 4;

        for i in 0..num_records {
            let rec_offset = 4 + i * record_size;
            if rec_offset + record_size > page.len() {
                break;
            }
            let pid = u32::from_le_bytes(page[rec_offset..rec_offset + 4].try_into().unwrap());
            if pid == point_id {
                let mut neighbors = Vec::new();
                for j in 0..compressed_max_degree as usize {
                    let noff = rec_offset + 4 + j * 4;
                    let nid = u32::from_le_bytes(page[noff..noff + 4].try_into().unwrap());
                    if nid == u32::MAX {
                        break;
                    }
                    neighbors.push(nid);
                }
                return neighbors;
            }
        }
        Vec::new()
    }

    /// Parse ALL points' compressed neighbors from a cluster page.
    /// Returns a map: point_id → compressed neighbors.
    pub fn parse_all_cluster_neighbors(
        page: &[u8],
        compressed_max_degree: u32,
    ) -> HashMap<u32, Vec<u32>> {
        let mut result = HashMap::new();
        if page.len() < 4 {
            return result;
        }
        let num_records = u32::from_le_bytes(page[0..4].try_into().unwrap()) as usize;
        let record_size = (1 + compressed_max_degree as usize) * 4;

        for i in 0..num_records {
            let rec_offset = 4 + i * record_size;
            if rec_offset + record_size > page.len() {
                break;
            }
            let pid = u32::from_le_bytes(page[rec_offset..rec_offset + 4].try_into().unwrap());
            let mut neighbors = Vec::new();
            for j in 0..compressed_max_degree as usize {
                let noff = rec_offset + 4 + j * 4;
                let nid = u32::from_le_bytes(page[noff..noff + 4].try_into().unwrap());
                if nid == u32::MAX {
                    break;
                }
                neighbors.push(nid);
            }
            result.insert(pid, neighbors);
        }
        result
    }

    // ─── Write ───

    /// Write the full index to a file.
    ///
    /// Each node's sector: [neighbor_page][data_page][cluster_page_bytes]
    /// The cluster_page_bytes should be pre-built via `build_cluster_page()`.
    pub fn write_to_file<P: AsRef<Path>>(
        &self,
        path: P,
        node_data: &[(Vec<u32>, Vec<f32>, Vec<u8>)], // (neighbors, vector, cluster_page_bytes)
    ) -> ANNResult<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        let sector_size = self.layout.node_sector_size();

        for (neighbors, vector, cluster_page_bytes) in node_data {
            let mut sector = vec![0u8; sector_size];

            // Neighbor page
            let mut offset = 0;
            let num_neighbors = neighbors.len() as u32;
            sector[offset..offset + 4].copy_from_slice(&num_neighbors.to_le_bytes());
            offset += 4;
            for &n in neighbors {
                sector[offset..offset + 4].copy_from_slice(&n.to_le_bytes());
                offset += 4;
            }

            // Data page
            offset = self.layout.neighbor_page_size();
            for &v in vector {
                sector[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
                offset += 4;
            }

            // Cluster page (redundant, same for all nodes in same cluster)
            let cluster_offset = self.layout.neighbor_page_size() + self.layout.data_page_size();
            let copy_len = cluster_page_bytes
                .len()
                .min(self.layout.cluster_page_size());
            sector[cluster_offset..cluster_offset + copy_len]
                .copy_from_slice(&cluster_page_bytes[..copy_len]);

            file.write_all(&sector)?;
        }

        file.flush()?;
        Ok(())
    }

    /// Build cluster page bytes from a ClusterPageData.
    ///
    /// Format:
    ///   [num_records: u32]
    ///   [record: point_id(u32) | neighbor_0(u32) | ... | neighbor_{max-1}(u32)] * num_records
    ///   [zero padding to cluster_page_size]
    pub fn build_cluster_page(&self, cluster: &ClusterPageData) -> Vec<u8> {
        let page_size = self.layout.cluster_page_size();
        let record_size = self.layout.cluster_record_size();
        let mut page = vec![0u8; page_size];

        let num_records = cluster.records.len() as u32;
        page[0..4].copy_from_slice(&num_records.to_le_bytes());

        for (i, (point_id, neighbors)) in cluster.records.iter().enumerate() {
            let rec_offset = 4 + i * record_size;
            if rec_offset + record_size > page.len() {
                break;
            }
            page[rec_offset..rec_offset + 4].copy_from_slice(&point_id.to_le_bytes());

            for j in 0..self.layout.compressed_max_degree as usize {
                let noff = rec_offset + 4 + j * 4;
                if j < neighbors.len() {
                    page[noff..noff + 4].copy_from_slice(&neighbors[j].to_le_bytes());
                } else {
                    // Pad with u32::MAX sentinel (not 0, since node 0 is a valid ID)
                    page[noff..noff + 4].copy_from_slice(&u32::MAX.to_le_bytes());
                }
            }
        }

        page
    }

    pub fn is_loaded(&self) -> bool {
        self.mmap.is_some()
    }

    pub fn compressed_max_degree(&self) -> u32 {
        self.layout.compressed_max_degree
    }

    pub fn layout(&self) -> &SectorLayout {
        &self.layout
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_roundtrip() {
        let layout = SectorLayout::new(4, 4, 2, 2, 5);
        let storage = MmapStorage::new(layout.clone());

        // Build cluster page for cluster 0 (both nodes)
        let cluster_page = storage.build_cluster_page(&ClusterPageData {
            cluster_id: 0,
            records: vec![(0, vec![1, 2]), (1, vec![0])],
        });

        // Per-node data: (neighbors, vector, cluster_page_bytes)
        let node_data = vec![
            (
                vec![1u32, 2, 3],
                vec![1.0f32, 2.0, 3.0, 4.0],
                cluster_page.clone(),
            ),
            (vec![0u32, 3], vec![5.0f32, 6.0, 7.0, 8.0], cluster_page),
        ];

        let tmp = std::env::temp_dir().join("mmap_cluster_storage_test.bin");
        storage.write_to_file(&tmp, &node_data).unwrap();

        let loaded = MmapStorage::open(&tmp, layout).unwrap();
        assert!(loaded.is_loaded());

        // Read neighbor page for node 0
        let nbr_page = loaded.read_neighbor_page(0).unwrap();
        let count = u32::from_le_bytes(nbr_page[0..4].try_into().unwrap());
        assert_eq!(count, 3);

        // Read data page for node 0
        let data_page = loaded.read_data_page(0).unwrap();
        let v0 = f32::from_le_bytes(data_page[0..4].try_into().unwrap());
        assert_eq!(v0, 1.0);

        // Read cluster page for node 0
        let cpage0 = loaded.read_cluster_neighbor_page(0).unwrap();
        let num_records = u32::from_le_bytes(cpage0[0..4].try_into().unwrap());
        assert_eq!(num_records, 2);

        // Parse point 0's neighbors
        let n0 = MmapStorage::parse_point_neighbors(cpage0, 0, 2);
        assert_eq!(n0, vec![1, 2]);

        // Parse point 1's neighbors
        let n1 = MmapStorage::parse_point_neighbors(cpage0, 1, 2);
        assert_eq!(n1, vec![0]);

        // Read cluster page for node 1 — same cluster page (redundant)
        let cpage1 = loaded.read_cluster_neighbor_page(1).unwrap();
        assert_eq!(cpage0, cpage1);

        // Parse all neighbors at once
        let all = MmapStorage::parse_all_cluster_neighbors(cpage0, 2);
        assert_eq!(all.len(), 2);
        assert_eq!(all[&0], vec![1, 2]);
        assert_eq!(all[&1], vec![0]);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_multiple_clusters() {
        let layout = SectorLayout::new(4, 4, 4, 3, 5);
        let storage = MmapStorage::new(layout.clone());

        // Cluster 0: nodes 0, 1
        let cluster0_page = storage.build_cluster_page(&ClusterPageData {
            cluster_id: 0,
            records: vec![(0, vec![2, 3]), (1, vec![3])],
        });

        // Cluster 1: nodes 2, 3
        let cluster1_page = storage.build_cluster_page(&ClusterPageData {
            cluster_id: 1,
            records: vec![(2, vec![0, 1]), (3, vec![0])],
        });

        let node_data = vec![
            (vec![1u32], vec![0.0f32; 4], cluster0_page.clone()),
            (vec![0u32], vec![1.0f32; 4], cluster0_page),
            (vec![3u32], vec![2.0f32; 4], cluster1_page.clone()),
            (vec![2u32], vec![3.0f32; 4], cluster1_page),
        ];

        let tmp = std::env::temp_dir().join("mmap_multi_cluster_test.bin");
        storage.write_to_file(&tmp, &node_data).unwrap();

        let loaded = MmapStorage::open(&tmp, layout).unwrap();

        // Node 0's cluster page (cluster 0)
        let page0 = loaded.read_cluster_neighbor_page(0).unwrap();
        let n0 = MmapStorage::parse_point_neighbors(page0, 0, 3);
        assert_eq!(n0, vec![2, 3]);
        let n1 = MmapStorage::parse_point_neighbors(page0, 1, 3);
        assert_eq!(n1, vec![3]);

        // Node 2's cluster page (cluster 1)
        let page2 = loaded.read_cluster_neighbor_page(2).unwrap();
        let n2 = MmapStorage::parse_point_neighbors(page2, 2, 3);
        assert_eq!(n2, vec![0, 1]);
        let n3 = MmapStorage::parse_point_neighbors(page2, 3, 3);
        assert_eq!(n3, vec![0]);

        // Nodes in same cluster have same cluster page
        let page1 = loaded.read_cluster_neighbor_page(1).unwrap();
        assert_eq!(page0, page1);
        let page3 = loaded.read_cluster_neighbor_page(3).unwrap();
        assert_eq!(page2, page3);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_combined_phase1_read() {
        let layout = SectorLayout::new(4, 4, 2, 2, 5);
        let storage = MmapStorage::new(layout.clone());

        let cluster_page = storage.build_cluster_page(&ClusterPageData {
            cluster_id: 0,
            records: vec![(0, vec![1]), (1, vec![0])],
        });

        let node_data = vec![
            (
                vec![1u32, 2],
                vec![1.0f32, 2.0, 3.0, 4.0],
                cluster_page.clone(),
            ),
            (vec![0u32], vec![5.0f32, 6.0, 7.0, 8.0], cluster_page),
        ];

        let tmp = std::env::temp_dir().join("mmap_combined_phase1_test.bin");
        storage.write_to_file(&tmp, &node_data).unwrap();

        let loaded = MmapStorage::open(&tmp, layout).unwrap();

        // Combined Phase 1 read: neighbor + data pages
        let (nbr_page, data_page) = loaded.read_neighbor_and_data_pages(0).unwrap();

        // Verify neighbor page content
        let count = u32::from_le_bytes(nbr_page[0..4].try_into().unwrap());
        assert_eq!(count, 2);
        let n0 = u32::from_le_bytes(nbr_page[4..8].try_into().unwrap());
        assert_eq!(n0, 1);

        // Verify data page content
        let v0 = f32::from_le_bytes(data_page[0..4].try_into().unwrap());
        assert_eq!(v0, 1.0);
        let v1 = f32::from_le_bytes(data_page[4..8].try_into().unwrap());
        assert_eq!(v1, 2.0);

        // Verify it matches individual reads
        let nbr_individual = loaded.read_neighbor_page(0).unwrap();
        let data_individual = loaded.read_data_page(0).unwrap();
        assert_eq!(nbr_page, nbr_individual);
        assert_eq!(data_page, data_individual);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_combined_phase2_read() {
        let layout = SectorLayout::new(4, 4, 2, 2, 5);
        let storage = MmapStorage::new(layout.clone());

        let cluster_page = storage.build_cluster_page(&ClusterPageData {
            cluster_id: 0,
            records: vec![(0, vec![1, 2]), (1, vec![0])],
        });

        let node_data = vec![
            (
                vec![1u32],
                vec![1.0f32, 2.0, 3.0, 4.0],
                cluster_page.clone(),
            ),
            (vec![0u32], vec![5.0f32, 6.0, 7.0, 8.0], cluster_page),
        ];

        let tmp = std::env::temp_dir().join("mmap_combined_phase2_test.bin");
        storage.write_to_file(&tmp, &node_data).unwrap();

        let loaded = MmapStorage::open(&tmp, layout).unwrap();

        // Combined Phase 2 read: data + cluster pages
        let (data_page, cluster_page) = loaded.read_data_and_cluster_pages(0).unwrap();

        // Verify data page
        let v0 = f32::from_le_bytes(data_page[0..4].try_into().unwrap());
        assert_eq!(v0, 1.0);

        // Verify cluster page
        let num_records = u32::from_le_bytes(cluster_page[0..4].try_into().unwrap());
        assert_eq!(num_records, 2);
        let neighbors = MmapStorage::parse_point_neighbors(cluster_page, 0, 2);
        assert_eq!(neighbors, vec![1, 2]);

        // Verify it matches individual reads
        let data_individual = loaded.read_data_page(0).unwrap();
        let cluster_individual = loaded.read_cluster_neighbor_page(0).unwrap();
        assert_eq!(data_page, data_individual);
        assert_eq!(cluster_page, cluster_individual);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_not_loaded() {
        let layout = SectorLayout::new(4, 4, 2, 2, 5);
        let storage = MmapStorage::new(layout);
        assert!(!storage.is_loaded());
        assert!(storage.read_neighbor_page(0).is_none());
        assert!(storage.read_cluster_neighbor_page(0).is_none());
    }

    #[test]
    fn test_parse_nonexistent_point() {
        let layout = SectorLayout::new(4, 4, 1, 2, 5);
        let storage = MmapStorage::new(layout.clone());

        let cluster_page = storage.build_cluster_page(&ClusterPageData {
            cluster_id: 0,
            records: vec![(5, vec![10, 20])],
        });

        let node_data = vec![(vec![1u32], vec![1.0f32; 4], cluster_page)];

        let tmp = std::env::temp_dir().join("mmap_nonexist_point_test.bin");
        storage.write_to_file(&tmp, &node_data).unwrap();

        let loaded = MmapStorage::open(&tmp, layout).unwrap();
        let page = loaded.read_cluster_neighbor_page(0).unwrap();

        // Point 5 exists
        assert_eq!(MmapStorage::parse_point_neighbors(page, 5, 2), vec![10, 20]);

        // Point 99 does not exist
        assert_eq!(
            MmapStorage::parse_point_neighbors(page, 99, 2),
            Vec::<u32>::new()
        );

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_node_zero_as_neighbor() {
        // Regression: node ID 0 must be a valid compressed neighbor,
        // not confused with zero-padding sentinel.
        let layout = SectorLayout::new(4, 4, 2, 3, 5);
        let storage = MmapStorage::new(layout.clone());

        // Point 1 has compressed neighbors [0, 2] — node 0 appears at j=0
        // Point 2 has compressed neighbors [3, 0] — node 0 appears at j=1
        let cluster_page = storage.build_cluster_page(&ClusterPageData {
            cluster_id: 0,
            records: vec![(1, vec![0, 2]), (2, vec![3, 0])],
        });

        let node_data = vec![
            (vec![1u32], vec![0.0f32; 4], cluster_page.clone()),
            (vec![0u32], vec![1.0f32; 4], cluster_page),
        ];

        let tmp = std::env::temp_dir().join("mmap_node_zero_test.bin");
        storage.write_to_file(&tmp, &node_data).unwrap();

        let loaded = MmapStorage::open(&tmp, layout).unwrap();
        let page = loaded.read_cluster_neighbor_page(0).unwrap();

        // Node 0 at position 0 should be preserved
        let n1 = MmapStorage::parse_point_neighbors(page, 1, 3);
        assert_eq!(
            n1,
            vec![0, 2],
            "Node 0 at j=0 must not be treated as padding"
        );

        // Node 0 at position 1 should also be preserved
        let n2 = MmapStorage::parse_point_neighbors(page, 2, 3);
        assert_eq!(
            n2,
            vec![3, 0],
            "Node 0 at j>0 must not be treated as padding"
        );

        // parse_all should also work correctly
        let all = MmapStorage::parse_all_cluster_neighbors(page, 3);
        assert_eq!(all[&1], vec![0, 2]);
        assert_eq!(all[&2], vec![3, 0]);

        std::fs::remove_file(&tmp).ok();
    }
}
