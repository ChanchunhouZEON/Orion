/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

/// Storage sector layout for Compressed DiskANN.
///
/// Per-node sector layout:
///   [neighbor_page]  — [count: u32][neighbor_ids: u32 * max_degree] (aligned to SECTOR_SIZE)
///   [data_page]      — [vector: f32 * dimension]                    (aligned to SECTOR_SIZE)
///   [cluster_page]   — per-node's same-cluster pruned neighbors     (aligned to SECTOR_SIZE)
///
/// The cluster_page stores pruned neighbor info for only the same-cluster nodes
/// that the current node connects to (not the entire cluster).
///
/// Cluster page format:
///   [num_records: u32]
///   [record 0: point_id(u32) | neighbor_0(u32) | ... | neighbor_{max-1}(u32)]
///   [record 1: point_id(u32) | neighbor_0(u32) | ... | neighbor_{max-1}(u32)]
///   ...
///   [padding to SECTOR_SIZE]
///
/// Each record's out-degree is padded with u32::MAX to compressed_max_degree if shorter.

pub const SECTOR_SIZE: usize = 4096; // 4KB pages

/// Layout for on-disk storage.
#[derive(Debug, Clone)]
pub struct SectorLayout {
    /// Maximum degree for the complete neighbor page.
    pub max_degree: u32,
    /// Dimension of the data vectors.
    pub dimension: u32,
    /// Number of nodes in the index.
    pub num_nodes: u32,
    /// Maximum compressed degree (= max_pruned_degree).
    pub compressed_max_degree: u32,
    /// Maximum number of records per cluster page (= max_same_cluster_edges + 1 for self).
    pub max_cluster_page_records: u32,
}

impl SectorLayout {
    pub fn new(
        max_degree: u32,
        dimension: u32,
        num_nodes: u32,
        compressed_max_degree: u32,
        max_cluster_page_records: u32,
    ) -> Self {
        Self {
            max_degree,
            dimension,
            num_nodes,
            compressed_max_degree,
            max_cluster_page_records,
        }
    }

    /// Size of the complete neighbor page (u32 count + u32 * max_degree neighbors).
    /// Aligned to SECTOR_SIZE.
    pub fn neighbor_page_size(&self) -> usize {
        let content = 4 + (self.max_degree as usize) * 4;
        align_to_sector(content)
    }

    /// Size of the data page (f32 * dimension). Aligned to SECTOR_SIZE.
    pub fn data_page_size(&self) -> usize {
        let content = (self.dimension as usize) * 4;
        align_to_sector(content)
    }

    /// Size of one record in a cluster page: (1 + compressed_max_degree) * 4 bytes.
    /// The record is: [point_id: u32][neighbor_0: u32]...[neighbor_{max-1}: u32]
    pub fn cluster_record_size(&self) -> usize {
        (1 + self.compressed_max_degree as usize) * 4
    }

    /// Maximum number of records that fit in one standard page (4KB).
    pub fn max_records_per_page(&self) -> usize {
        let record_size = self.cluster_record_size();
        if record_size == 0 {
            return 0;
        }
        (SECTOR_SIZE - 4) / record_size
    }

    /// Size of the cluster page (sector-aligned), based on max_cluster_page_records.
    /// This is the fixed size allocated for the cluster page in each node's sector.
    pub fn cluster_page_size(&self) -> usize {
        let max_per_page = self.max_records_per_page();
        if max_per_page == 0 {
            return SECTOR_SIZE;
        }
        // +1 for self record
        let total_records = self.max_cluster_page_records as usize + 1;
        let pages_needed = (total_records + max_per_page - 1) / max_per_page;
        pages_needed * SECTOR_SIZE
    }

    /// Size of a single node's sector (neighbor_page + data_page + cluster_page).
    pub fn node_sector_size(&self) -> usize {
        self.neighbor_page_size() + self.data_page_size() + self.cluster_page_size()
    }

    /// Offset of a node's sector in the storage file.
    pub fn node_offset(&self, node_id: u32) -> u64 {
        (node_id as u64) * (self.node_sector_size() as u64)
    }
}

/// Align a byte count up to the next SECTOR_SIZE boundary.
pub fn align_to_sector(size: usize) -> usize {
    (size + SECTOR_SIZE - 1) / SECTOR_SIZE * SECTOR_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neighbor_page_aligned_to_sector() {
        let layout = SectorLayout::new(32, 128, 100, 12, 4);
        let size = layout.neighbor_page_size();
        assert_eq!(size % SECTOR_SIZE, 0);
    }

    #[test]
    fn test_data_page_aligned_to_sector() {
        let layout = SectorLayout::new(32, 128, 100, 12, 4);
        let size = layout.data_page_size();
        assert_eq!(size % SECTOR_SIZE, 0);
    }

    #[test]
    fn test_cluster_record_size() {
        let layout = SectorLayout::new(32, 128, 100, 12, 4);
        // (1+12)*4 = 52 bytes
        assert_eq!(layout.cluster_record_size(), 52);
    }

    #[test]
    fn test_max_records_per_page() {
        let layout = SectorLayout::new(32, 128, 100, 12, 4);
        // (4096 - 4) / 52 = 78
        assert_eq!(layout.max_records_per_page(), 78);
    }

    #[test]
    fn test_cluster_fits_in_one_page() {
        // max_cluster_page_records=4, so total records = 4+1 = 5
        // compressed_max_degree=12, record_size=52
        // 5 records of 52 bytes = 260 + 4 header = 264 bytes << 4096
        let layout = SectorLayout::new(32, 128, 100, 12, 4);
        assert_eq!(layout.cluster_page_size(), SECTOR_SIZE);
    }

    #[test]
    fn test_node_sector_is_sum_of_pages() {
        let layout = SectorLayout::new(32, 128, 100, 12, 4);
        assert_eq!(
            layout.node_sector_size(),
            layout.neighbor_page_size() + layout.data_page_size() + layout.cluster_page_size()
        );
    }

    #[test]
    fn test_node_offsets_sequential() {
        let layout = SectorLayout::new(32, 128, 100, 12, 4);
        let sector_size = layout.node_sector_size();
        assert_eq!(layout.node_offset(0), 0);
        assert_eq!(layout.node_offset(1), sector_size as u64);
        assert_eq!(layout.node_offset(2), 2 * sector_size as u64);
    }

    #[test]
    fn test_small_degree_fits_one_sector() {
        let layout = SectorLayout::new(4, 4, 10, 2, 4);
        assert_eq!(layout.neighbor_page_size(), SECTOR_SIZE);
    }
}
