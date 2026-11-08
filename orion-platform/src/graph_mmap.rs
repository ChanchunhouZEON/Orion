/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use memmap2::{Mmap, MmapOptions};
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

/// Magic number for the ANNS mmap graph format: "ANNS" in little-endian.
pub const MMAP_MAGIC: u32 = 0x414E4E53;
/// Current format version.
pub const MMAP_VERSION: u32 = 1;

/// Header size in bytes (fixed 32 bytes).
pub const HEADER_SIZE: usize = 32;

/// Algorithm identifiers for the graph format.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmId {
    Generic = 0,
    Hnsw = 1,
    Nsg = 2,
    DiskAnn = 3,
    Orion = 4,
}

impl TryFrom<u8> for AlgorithmId {
    type Error = io::Error;
    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Generic),
            1 => Ok(Self::Hnsw),
            2 => Ok(Self::Nsg),
            3 => Ok(Self::DiskAnn),
            4 => Ok(Self::Orion),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown algorithm id: {v}"),
            )),
        }
    }
}

/// Fixed 32-byte header for the mmap graph file.
///
/// Layout:
/// ```text
/// [0..4]   magic: u32 = 0x414E4E53 ("ANNS")
/// [4..8]   version: u32 = 1
/// [8..12]  num_nodes: u32
/// [12..16] max_degree: u32
/// [16..20] dimension: u32
/// [20]     metric: u8 (0=L2, 1=Cosine)
/// [21]     algorithm: u8
/// [22..32] reserved: [u8; 10]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct GraphHeader {
    pub num_nodes: u32,
    pub max_degree: u32,
    pub dimension: u32,
    pub metric: u8,
    pub algorithm: AlgorithmId,
}

impl GraphHeader {
    /// Per-node stride in the adjacency section: (1 + max_degree) * 4 bytes.
    /// First u32 = actual neighbor count, followed by `max_degree` u32 neighbor ids (padded).
    pub fn node_adj_stride(&self) -> usize {
        (1 + self.max_degree as usize) * 4
    }

    /// Per-node stride in the vector section: dimension * 4 bytes (f32).
    pub fn node_vec_stride(&self) -> usize {
        self.dimension as usize * 4
    }
}

/// Read-only memory-mapped graph for zero-copy access.
///
/// File layout:
/// ```text
/// [Header: 32 bytes]
/// [Algorithm-specific header: algo_header_size bytes]
/// [Graph adjacency data: num_nodes * (1 + max_degree) * 4 bytes]
/// [Vector data: num_nodes * dimension * 4 bytes]
/// ```
pub struct MmapGraph {
    mmap: Mmap,
    pub header: GraphHeader,
    algo_header_size: usize,
}

impl MmapGraph {
    /// Open a memory-mapped graph file.
    ///
    /// `algo_header_size` specifies the size of the algorithm-specific header
    /// that sits between the common header and the adjacency data.
    pub fn open<P: AsRef<Path>>(path: P, algo_header_size: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File too small for header",
            ));
        }

        // Parse header
        let magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if magic != MMAP_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid magic: 0x{magic:08X}, expected 0x{MMAP_MAGIC:08X}"),
            ));
        }

        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());
        if version != MMAP_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported version: {version}"),
            ));
        }

        let num_nodes = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let max_degree = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
        let dimension = u32::from_le_bytes(mmap[16..20].try_into().unwrap());
        let metric = mmap[20];
        let algorithm = AlgorithmId::try_from(mmap[21])?;

        let header = GraphHeader {
            num_nodes,
            max_degree,
            dimension,
            metric,
            algorithm,
        };

        let expected_size = HEADER_SIZE
            + algo_header_size
            + (num_nodes as usize) * header.node_adj_stride()
            + (num_nodes as usize) * header.node_vec_stride();

        if mmap.len() < expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "File too small: {} bytes, expected at least {} bytes",
                    mmap.len(),
                    expected_size
                ),
            ));
        }

        Ok(Self {
            mmap,
            header,
            algo_header_size,
        })
    }

    /// Raw bytes of the algorithm-specific header section.
    pub fn algo_header_bytes(&self) -> &[u8] {
        &self.mmap[HEADER_SIZE..HEADER_SIZE + self.algo_header_size]
    }

    /// Offset where adjacency data starts.
    fn adj_offset(&self) -> usize {
        HEADER_SIZE + self.algo_header_size
    }

    /// Offset where vector data starts.
    fn vec_offset(&self) -> usize {
        self.adj_offset() + (self.header.num_nodes as usize) * self.header.node_adj_stride()
    }

    /// Get the neighbors of a node as a zero-copy slice. Returns `(count, &[u32])`.
    pub fn neighbors(&self, node_id: u32) -> &[u32] {
        let stride = self.header.node_adj_stride();
        let base = self.adj_offset() + (node_id as usize) * stride;
        let count = u32::from_le_bytes(self.mmap[base..base + 4].try_into().unwrap()) as usize;
        let count = count.min(self.header.max_degree as usize);
        let ptr = &self.mmap[base + 4..base + 4 + count * 4];
        // SAFETY: u32 is 4-byte aligned and the slice length is correct.
        // On little-endian platforms (x86, ARM), this is zero-copy.
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u32, count) }
    }

    /// Get the vector for a node as a zero-copy f32 slice.
    pub fn vector_f32(&self, node_id: u32) -> &[f32] {
        let stride = self.header.node_vec_stride();
        let base = self.vec_offset() + (node_id as usize) * stride;
        let dim = self.header.dimension as usize;
        let ptr = &self.mmap[base..base + dim * 4];
        unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, dim) }
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> u32 {
        self.header.num_nodes
    }
}

/// Writer for creating mmap-compatible graph files.
pub struct GraphWriter {
    header: GraphHeader,
    algo_header: Vec<u8>,
}

impl GraphWriter {
    pub fn new(header: GraphHeader) -> Self {
        Self {
            header,
            algo_header: Vec::new(),
        }
    }

    pub fn set_algo_header(&mut self, data: Vec<u8>) {
        self.algo_header = data;
    }

    /// Write the graph to a file.
    ///
    /// `adj_data`: closure that returns (num_neighbors, neighbors_slice) for each node
    /// `vec_data`: closure that returns f32 slice for each node's vector
    pub fn write<P, FA, FV>(&self, path: P, adj_data: FA, vec_data: FV) -> io::Result<()>
    where
        P: AsRef<Path>,
        FA: Fn(u32) -> (u32, Vec<u32>),
        FV: Fn(u32) -> Vec<f32>,
    {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;

        // Write common header (32 bytes)
        let mut header_bytes = [0u8; HEADER_SIZE];
        header_bytes[0..4].copy_from_slice(&MMAP_MAGIC.to_le_bytes());
        header_bytes[4..8].copy_from_slice(&MMAP_VERSION.to_le_bytes());
        header_bytes[8..12].copy_from_slice(&self.header.num_nodes.to_le_bytes());
        header_bytes[12..16].copy_from_slice(&self.header.max_degree.to_le_bytes());
        header_bytes[16..20].copy_from_slice(&self.header.dimension.to_le_bytes());
        header_bytes[20] = self.header.metric;
        header_bytes[21] = self.header.algorithm as u8;
        // [22..32] reserved = 0
        file.write_all(&header_bytes)?;

        // Write algorithm-specific header
        file.write_all(&self.algo_header)?;

        // Write adjacency data (fixed stride per node)
        let stride = self.header.node_adj_stride();
        let mut adj_buf = vec![0u8; stride];
        for node_id in 0..self.header.num_nodes {
            adj_buf.fill(0);
            let (count, neighbors) = adj_data(node_id);
            let count = count.min(self.header.max_degree);
            adj_buf[0..4].copy_from_slice(&count.to_le_bytes());
            for (i, &nid) in neighbors.iter().take(count as usize).enumerate() {
                let off = 4 + i * 4;
                adj_buf[off..off + 4].copy_from_slice(&nid.to_le_bytes());
            }
            file.write_all(&adj_buf)?;
        }

        // Write vector data (contiguous f32 arrays)
        for node_id in 0..self.header.num_nodes {
            let vec = vec_data(node_id);
            for &v in &vec {
                file.write_all(&v.to_le_bytes())?;
            }
        }

        file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read_roundtrip() {
        let dir = std::env::temp_dir().join("anns_mmap_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_graph.anns");

        let header = GraphHeader {
            num_nodes: 5,
            max_degree: 3,
            dimension: 4,
            metric: 0, // L2
            algorithm: AlgorithmId::Generic,
        };

        // Create adjacency: node i connects to (i+1)%5 and (i+2)%5
        let writer = GraphWriter::new(header);
        writer
            .write(
                &path,
                |node_id| {
                    let n1 = (node_id + 1) % 5;
                    let n2 = (node_id + 2) % 5;
                    (2, vec![n1, n2])
                },
                |node_id| vec![node_id as f32; 4],
            )
            .unwrap();

        // Read back
        let graph = MmapGraph::open(&path, 0).unwrap();
        assert_eq!(graph.header.num_nodes, 5);
        assert_eq!(graph.header.max_degree, 3);
        assert_eq!(graph.header.dimension, 4);

        for i in 0..5u32 {
            let neighbors = graph.neighbors(i);
            assert_eq!(neighbors.len(), 2);
            assert_eq!(neighbors[0], (i + 1) % 5);
            assert_eq!(neighbors[1], (i + 2) % 5);

            let vec = graph.vector_f32(i);
            assert_eq!(vec.len(), 4);
            assert_eq!(vec[0], i as f32);
        }

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_algo_header() {
        let dir = std::env::temp_dir().join("anns_mmap_test_algo");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_algo.anns");

        let header = GraphHeader {
            num_nodes: 2,
            max_degree: 2,
            dimension: 3,
            metric: 0,
            algorithm: AlgorithmId::Nsg,
        };

        let mut writer = GraphWriter::new(header);
        // 8-byte algo header: navigation_node (u32) + r (u32)
        let mut algo = vec![0u8; 8];
        algo[0..4].copy_from_slice(&42u32.to_le_bytes());
        algo[4..8].copy_from_slice(&10u32.to_le_bytes());
        writer.set_algo_header(algo);

        writer
            .write(&path, |_| (1, vec![0]), |node_id| vec![node_id as f32; 3])
            .unwrap();

        let graph = MmapGraph::open(&path, 8).unwrap();
        let algo_bytes = graph.algo_header_bytes();
        let nav = u32::from_le_bytes(algo_bytes[0..4].try_into().unwrap());
        let r = u32::from_le_bytes(algo_bytes[4..8].try_into().unwrap());
        assert_eq!(nav, 42);
        assert_eq!(r, 10);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_invalid_magic() {
        let dir = std::env::temp_dir().join("anns_mmap_test_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad_magic.anns");

        std::fs::write(&path, &[0u8; 32]).unwrap();
        assert!(MmapGraph::open(&path, 0).is_err());

        std::fs::remove_dir_all(&dir).ok();
    }
}
