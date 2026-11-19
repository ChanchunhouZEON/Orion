/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Native cache and STAG readers validate records before publishing the slab.
use super::*;
use orion_ann_platform::buffered_io::{read_exact_in_blocks, write_in_blocks};
use std::io::{BufWriter, Read, Write};

impl PhasedGraph {
    /// Writes the graph header and contiguous slab using the common block writer.
    ///
    /// The caller owns flushing, durability and publication of the resulting file.
    pub(crate) fn encode(&self, writer: &mut BufWriter<impl Write>) -> Result<()> {
        writer.write_all(bytemuck::cast_slice(&self.file_header()))?;
        write_in_blocks(writer, self.buffer_bytes())?;
        Ok(())
    }

    /// Returns the native graph header describing node count, degree and slab stride.
    fn file_header(&self) -> [u32; 4] {
        [
            self.num_nodes as u32,
            self.max_degree,
            self.stride as u32,
            self.base_local_count,
        ]
    }

    /// Read the existing native cache format. Counts, IDs and file length are
    /// checked before the graph can be searched. Capacity includes cache-line
    /// padding because this format does not store the original extras limit.
    ///
    /// # Errors
    /// Rejects malformed/truncated records and allocation or filesystem failures.
    #[cfg(test)]
    pub(crate) fn decode(reader: impl Read, file_bytes: u64) -> Result<Self> {
        Self::decode_with_storage(reader, file_bytes, &Default::default())
    }

    /// Validates the native graph header, reads the slab into planned storage and checks every row.
    pub(crate) fn decode_with_storage(
        mut reader: impl Read,
        file_bytes: u64,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        let mut header = [0u32; 4];
        reader.read_exact(bytemuck::cast_slice_mut(&mut header))?;
        let [nodes, max_degree, stride, base_local_count] = header;
        let (num_nodes, stride) = (nodes as usize, stride as usize);
        let total = num_nodes
            .checked_mul(stride)
            .ok_or_else(|| Error::invalid_data("graph size overflow"))?;
        let expected = total
            .checked_mul(4)
            .and_then(|n| n.checked_add(16))
            .ok_or_else(|| Error::invalid_data("graph byte size overflow"))?;
        if num_nodes == 0
            || nodes == u32::MAX
            || max_degree == 0
            || stride < HEADER_U32
            || stride % (CACHE_LINE_BYTES / 4) != 0
            || max_degree as usize > stride - HEADER_U32
            || base_local_count > max_degree
            || expected as u64 != file_bytes
        {
            return Err(
                Error::invalid_data("invalid graph header or payload length")
                    .at("load_graph")
                    .with_detail("expected_bytes", expected)
                    .with_detail("actual_bytes", file_bytes),
            );
        }
        let mut graph = Self::allocate_with_storage(
            num_nodes,
            max_degree,
            stride - HEADER_U32 - max_degree as usize,
            stride,
            base_local_count,
            storage,
        )?;
        // Bound destination spans as well as buffering: BufReader may otherwise
        // submit the entire mapped slab in one underlying read request.
        if storage.reclaim_after_write {
            // Flush completed windows before allocating further dirty file pages.
            // Validation still runs on the complete graph before publication.
            let window = orion_ann_platform::buffered_io::RECLAIM_WINDOW_BYTES / size_of::<u32>();
            for start in (0..total).step_by(window) {
                let end = (start + window).min(total);
                read_exact_in_blocks(
                    &mut reader,
                    bytemuck::cast_slice_mut(&mut graph.buffer[start..end]),
                )?;
                graph.buffer.flush_and_reclaim(start * 4..end * 4)?;
            }
        } else {
            read_exact_in_blocks(
                &mut reader,
                bytemuck::cast_slice_mut(graph.buffer.as_mut_slice()),
            )?;
        }
        graph.validate_rows()?;
        if storage.reclaim_after_write {
            graph.buffer.flush_and_reclaim(0..total * 4)?;
        }
        Ok(graph)
    }

    /// Import STAG v3 directly into the final slab, retaining one node's IDs.
    ///
    /// # Errors
    /// Rejects invalid headers, counts, IDs, truncated records or trailing bytes.
    #[cfg(test)]
    pub(crate) fn decode_staged(
        reader: impl Read,
        file_bytes: u64,
        expected_nodes: usize,
        expected_degree: u32,
    ) -> Result<Self> {
        Self::decode_staged_with_storage(
            reader,
            file_bytes,
            expected_nodes,
            expected_degree,
            &Default::default(),
        )
    }

    /// Streams STAG v3 rows into an aligned slab, rejecting incompatible shape, capacities or trailing data.
    pub(crate) fn decode_staged_with_storage(
        mut reader: impl Read,
        file_bytes: u64,
        expected_nodes: usize,
        expected_degree: u32,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        /// Reads one little-endian STAG header or row-count word, propagating truncation errors.
        fn word(reader: &mut impl Read) -> Result<u32> {
            let mut bytes = [0; 4];
            reader.read_exact(&mut bytes)?;
            Ok(u32::from_le_bytes(bytes))
        }
        if word(&mut reader)? != 0x53544147 || word(&mut reader)? != 3 {
            return Err(Error::invalid_data("expected STAG v3"));
        }
        let n = word(&mut reader)? as usize;
        let degree = word(&mut reader)?;
        let extra = word(&mut reader)? as usize;
        let entry = word(&mut reader)? as usize;
        if n == 0
            || n != expected_nodes
            || entry >= n
            || degree != expected_degree
            || degree as usize > n
            || extra > n
            || file_bytes < 24 + n as u64 * 12
        {
            return Err(Error::invalid_data(
                "invalid staged header, shape mismatch or truncated records",
            ));
        }
        let mut graph = Self::try_empty_with_storage(n, degree, extra, storage)?;
        let mut ids = Vec::new();
        for node in 0..n {
            let lc = word(&mut reader)? as usize;
            let rc = word(&mut reader)? as usize;
            let ec = word(&mut reader)? as usize;
            let neighbors = lc
                .checked_add(rc)
                .ok_or_else(|| Error::invalid_data("degree overflow"))?;
            let count = neighbors
                .checked_add(ec)
                .ok_or_else(|| Error::invalid_data("record size overflow"))?;
            if neighbors > degree as usize || ec > extra {
                return Err(Error::invalid_data(
                    "staged record exceeds header capacities",
                ));
            }
            ids.clear();
            for _ in 0..count {
                let id = word(&mut reader)?;
                if id as usize >= n {
                    return Err(Error::invalid_data("staged neighbor ID outside graph"));
                }
                ids.push(id);
            }
            // Exclusive construction: no lock acquisition and no second graph.
            let row = &mut graph.buffer[node * graph.stride..(node + 1) * graph.stride];
            Self::write_row(row, &ids[..lc], &ids[lc..neighbors], &ids[neighbors..]);
            if node > 0 && node % 10_000_000 == 0 {
                log::info!("Imported {node}/{n} staged nodes");
            }
        }
        let mut tail = [0];
        if reader.read(&mut tail)? != 0 {
            return Err(Error::invalid_data("trailing bytes after staged graph"));
        }
        Ok(graph)
    }
}
