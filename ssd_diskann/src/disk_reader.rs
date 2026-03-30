use crate::disk_layout::{DiskIndexMetadata, SECTOR_SIZE};
use byteorder::{ByteOrder, LittleEndian};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Zero-copy disk index reader backed by mmap.
pub struct DiskReader {
    mmap: Mmap,
    pub meta: DiskIndexMetadata,
}

/// A node read from disk: vector bytes and neighbor IDs.
pub struct DiskNode<'a> {
    pub vector_bytes: &'a [u8],
    pub neighbors: Vec<u32>,
}

impl DiskReader {
    /// Open a disk index file for reading.
    pub fn open<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Parse sector 0 metadata
        anyhow::ensure!(
            mmap.len() >= SECTOR_SIZE,
            "File too small for metadata sector"
        );
        let s = &mmap[..DiskIndexMetadata::HEADER_SIZE];
        let meta = DiskIndexMetadata {
            num_points: LittleEndian::read_u64(&s[0..8]),
            dimension: LittleEndian::read_u64(&s[8..16]),
            medoid: LittleEndian::read_u64(&s[16..24]),
            max_node_len: LittleEndian::read_u64(&s[24..32]),
            nodes_per_sector: LittleEndian::read_u64(&s[32..40]),
            max_degree: LittleEndian::read_u64(&s[40..48]),
        };

        Ok(Self { mmap, meta })
    }

    /// Read a single node (vector bytes + neighbor IDs) from the disk index.
    pub fn read_node(&self, node_id: u32) -> Option<DiskNode<'_>> {
        let nid = node_id as u64;
        if nid >= self.meta.num_points {
            return None;
        }

        let nps = self.meta.nodes_per_sector;
        let mnl = self.meta.max_node_len as usize;
        let dim = self.meta.dimension as usize;

        // Which sector and slot?
        let sector_idx = nid / nps;
        let slot = (nid % nps) as usize;

        // Sector 0 is metadata, data starts at sector 1
        let sector_offset = ((sector_idx + 1) as usize) * SECTOR_SIZE;
        let node_offset = sector_offset + slot * mnl;

        if node_offset + mnl > self.mmap.len() {
            return None;
        }

        let node_bytes = &self.mmap[node_offset..node_offset + mnl];

        // Vector: first dim*4 bytes
        let vec_len = dim * 4;
        let vector_bytes = &node_bytes[..vec_len];

        // num_nbrs: u32 at offset vec_len
        let num_nbrs = LittleEndian::read_u32(&node_bytes[vec_len..vec_len + 4]) as usize;
        let num_nbrs = num_nbrs.min(self.meta.max_degree as usize);

        // neighbor IDs: starting at vec_len + 4
        let nbr_start = vec_len + 4;
        let mut neighbors = Vec::with_capacity(num_nbrs);
        for i in 0..num_nbrs {
            let off = nbr_start + i * 4;
            neighbors.push(LittleEndian::read_u32(&node_bytes[off..off + 4]));
        }

        Some(DiskNode {
            vector_bytes,
            neighbors,
        })
    }

    /// Parse an f32 vector from raw bytes.
    pub fn parse_vector(bytes: &[u8], dim: usize) -> Vec<f32> {
        let mut vec = Vec::with_capacity(dim);
        for i in 0..dim {
            vec.push(LittleEndian::read_f32(&bytes[i * 4..(i + 1) * 4]));
        }
        vec
    }

    /// Parse an f32 vector from raw bytes into a fixed-size array.
    pub fn parse_vector_array<const N: usize>(bytes: &[u8]) -> [f32; N] {
        let mut arr = [0.0f32; N];
        for i in 0..N {
            arr[i] = LittleEndian::read_f32(&bytes[i * 4..(i + 1) * 4]);
        }
        arr
    }

    /// Get the medoid (entry point) node ID.
    pub fn medoid(&self) -> u32 {
        self.meta.medoid as u32
    }

    /// Get the dimension of vectors.
    pub fn dimension(&self) -> usize {
        self.meta.dimension as usize
    }

    /// Get the number of points.
    pub fn num_points(&self) -> usize {
        self.meta.num_points as usize
    }

    /// Prefetch a node's sector into OS page cache.
    pub fn prefetch_node(&self, node_id: u32) {
        let nid = node_id as u64;
        if nid >= self.meta.num_points {
            return;
        }
        let nps = self.meta.nodes_per_sector;
        let sector_idx = nid / nps;
        let sector_offset = ((sector_idx + 1) as usize) * SECTOR_SIZE;
        if sector_offset + SECTOR_SIZE <= self.mmap.len() {
            // Touch the first byte of the sector to trigger page fault / prefetch
            let _ = self.mmap[sector_offset];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vector() {
        let mut bytes = vec![0u8; 8];
        LittleEndian::write_f32(&mut bytes[0..4], 1.0);
        LittleEndian::write_f32(&mut bytes[4..8], 2.0);
        let v = DiskReader::parse_vector(&bytes, 2);
        assert_eq!(v, vec![1.0, 2.0]);
    }

    #[test]
    fn test_parse_vector_array() {
        let mut bytes = vec![0u8; 8];
        LittleEndian::write_f32(&mut bytes[0..4], 3.0);
        LittleEndian::write_f32(&mut bytes[4..8], 4.0);
        let arr: [f32; 2] = DiskReader::parse_vector_array(&bytes);
        assert_eq!(arr, [3.0, 4.0]);
    }
}
