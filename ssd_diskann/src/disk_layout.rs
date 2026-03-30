use byteorder::{LittleEndian, WriteBytesExt};
use diskann::model::InMemoryGraph;
use ndarray::ArcArray2;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Disk index sector size (4KB).
pub const SECTOR_SIZE: usize = 4096;

/// Metadata stored in sector 0 of the disk index.
#[derive(Debug, Clone)]
pub struct DiskIndexMetadata {
    pub num_points: u64,
    pub dimension: u64,
    pub medoid: u64,
    pub max_node_len: u64,
    pub nodes_per_sector: u64,
    pub max_degree: u64,
}

impl DiskIndexMetadata {
    /// Byte size of the serialized metadata header (6 x u64 = 48 bytes).
    pub const HEADER_SIZE: usize = 6 * 8;
}

/// Compute the max node byte length for a given dimension and degree.
/// Node layout: [f32_vector: dim*4][num_nbrs: u32][neighbor_ids: u32*degree]
pub fn max_node_len(dimension: usize, degree: usize) -> usize {
    dimension * 4 + 4 + degree * 4
}

/// How many nodes fit in one sector (excluding sector 0 metadata).
pub fn nodes_per_sector(max_node_len: usize) -> usize {
    if max_node_len == 0 {
        return 0;
    }
    SECTOR_SIZE / max_node_len
}

/// Write the disk index to a file.
///
/// Format:
/// - Sector 0: metadata (DiskIndexMetadata)
/// - Sector 1+: packed nodes, `nodes_per_sector` nodes per sector
/// - Each node: [f32_vector][num_nbrs: u32][neighbor_ids: u32[]]
///   padded to max_node_len bytes
pub fn write_disk_index<P: AsRef<Path>>(
    path: P,
    data: &ArcArray2<f32>,
    graph: &InMemoryGraph,
    medoid: u32,
) -> anyhow::Result<DiskIndexMetadata> {
    let num_points = data.nrows();
    let dimension = data.ncols();
    let degree = graph.max_degree() as usize;

    let mnl = max_node_len(dimension, degree);
    let nps = nodes_per_sector(mnl);
    anyhow::ensure!(nps > 0, "Node too large for sector: max_node_len={mnl}");

    let meta = DiskIndexMetadata {
        num_points: num_points as u64,
        dimension: dimension as u64,
        medoid: medoid as u64,
        max_node_len: mnl as u64,
        nodes_per_sector: nps as u64,
        max_degree: degree as u64,
    };

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // ─── Sector 0: metadata ───
    let mut sector0 = vec![0u8; SECTOR_SIZE];
    {
        let mut cursor = &mut sector0[..];
        cursor.write_u64::<LittleEndian>(meta.num_points)?;
        cursor.write_u64::<LittleEndian>(meta.dimension)?;
        cursor.write_u64::<LittleEndian>(meta.medoid)?;
        cursor.write_u64::<LittleEndian>(meta.max_node_len)?;
        cursor.write_u64::<LittleEndian>(meta.nodes_per_sector)?;
        cursor.write_u64::<LittleEndian>(meta.max_degree)?;
    }
    w.write_all(&sector0)?;

    // ─── Sector 1+: node data ───
    let total_sectors = (num_points + nps - 1) / nps;
    let mut node_buf = vec![0u8; mnl];

    for sector_idx in 0..total_sectors {
        let mut sector = vec![0u8; SECTOR_SIZE];
        let base_node = sector_idx * nps;

        for slot in 0..nps {
            let node_id = base_node + slot;
            if node_id >= num_points {
                break;
            }

            // Clear node buffer
            node_buf.iter_mut().for_each(|b| *b = 0);
            let mut offset = 0;

            // Write vector (f32 * dim)
            let row = data.row(node_id);
            for &val in row.iter() {
                let bytes = val.to_le_bytes();
                node_buf[offset..offset + 4].copy_from_slice(&bytes);
                offset += 4;
            }

            // Write num_nbrs
            let neighbors = graph
                .read_vertex_and_neighbors(node_id as u32)
                .map_err(|e| anyhow::anyhow!("Failed to read node {node_id}: {e:?}"))?;
            let nbrs = neighbors.get_neighbors();
            let num_nbrs = nbrs.len() as u32;
            node_buf[offset..offset + 4].copy_from_slice(&num_nbrs.to_le_bytes());
            offset += 4;

            // Write neighbor IDs
            for &nbr in nbrs.iter() {
                node_buf[offset..offset + 4].copy_from_slice(&nbr.to_le_bytes());
                offset += 4;
            }

            // Copy into sector at the right slot position
            let slot_offset = slot * mnl;
            sector[slot_offset..slot_offset + mnl].copy_from_slice(&node_buf);
        }

        w.write_all(&sector)?;
    }

    w.flush()?;
    log::info!(
        "Wrote disk index: {} points, dim={}, degree={}, {} sectors, max_node_len={}, nodes_per_sector={}",
        num_points, dimension, degree, total_sectors + 1, mnl, nps
    );

    Ok(meta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_node_len() {
        // SIFT-128, degree 32: 128*4 + 4 + 32*4 = 512 + 4 + 128 = 644
        assert_eq!(max_node_len(128, 32), 644);
    }

    #[test]
    fn test_nodes_per_sector() {
        // 4096 / 644 = 6
        assert_eq!(nodes_per_sector(644), 6);
    }

    #[test]
    fn test_metadata_header_size() {
        assert_eq!(DiskIndexMetadata::HEADER_SIZE, 48);
    }
}
