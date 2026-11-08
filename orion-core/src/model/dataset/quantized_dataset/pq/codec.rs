/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Explicit little-endian payload. Runtime authenticates the stream before
//! publishing it; decoding validates shape/lengths before allocating.
use super::{FixedChunkPQTable, PqBuildOptions, PqCodeStorage, PqDataset, storage::Codes};
use crate::{Error, Result};
use byteorder::{LittleEndian as LE, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};
impl PqDataset {
    #[doc(hidden)]
    pub fn encode(&self, w: &mut dyn Write) -> Result<()> {
        w.write_all(b"ORIONPQ\0")?;
        w.write_u64::<LE>(self.rows as u64)?;
        w.write_u64::<LE>(self.dimension() as u64)?;
        w.write_u64::<LE>(self.table.get_num_chunks() as u64)?;
        for &x in self.table.offsets.iter() {
            w.write_u64::<LE>(x as u64)?;
        }
        write_floats(w, &self.table.mean)?;
        write_floats(w, &self.table.centers)?;
        w.write_all(self.codes())?;
        Ok(())
    }
    #[doc(hidden)]
    pub fn decode(r: &mut dyn Read, bytes: u64) -> Result<Self> {
        Self::decode_with_storage(r, bytes, PqCodeStorage::Resident)
    }
    /// Decode a payload whose integrity is checked by the caller's object store.
    /// # Errors
    /// Rejects corrupt layouts, nonfinite centers, overflow and truncated payloads.
    /// On-demand mode copies to a private temporary file before mapping it.
    pub fn decode_with_storage(
        r: &mut dyn Read,
        bytes: u64,
        storage: PqCodeStorage,
    ) -> Result<Self> {
        Self::decode_with_options(r, bytes, &storage.options())
    }
    /// Decode with a selected private mapping directory.
    /// # Errors
    /// Same checked wire-format contract as `decode_with_storage`.
    pub fn decode_with_options(
        r: &mut dyn Read,
        bytes: u64,
        storage: &orion_ann_platform::region::RegionOptions,
    ) -> Result<Self> {
        let bad = || Error::cache("invalid PQ payload");
        let mut magic = [0; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"ORIONPQ\0" {
            return Err(bad());
        }
        let rows = usize::try_from(r.read_u64::<LE>()?).map_err(|_| bad())?;
        let dim = usize::try_from(r.read_u64::<LE>()?).map_err(|_| bad())?;
        let m = usize::try_from(r.read_u64::<LE>()?).map_err(|_| bad())?;
        PqBuildOptions {
            chunks: Some(m),
            ..Default::default()
        }
        .chunks(dim)?;
        if rows == 0 || rows >= u32::MAX as usize {
            return Err(bad());
        }
        let expected = 32usize
            .checked_add((m + 1) * 8)
            .and_then(|v| v.checked_add(dim.checked_mul(257 * 4)?))
            .and_then(|v| v.checked_add(rows.checked_mul(m)?))
            .ok_or_else(bad)?;
        if expected as u64 != bytes {
            return Err(bad());
        }
        let offsets = (0..=m)
            .map(|_| {
                usize::try_from(r.read_u64::<LE>()?)
                    .map_err(|_| std::io::Error::other("PQ offset overflow"))
            })
            .collect::<std::io::Result<Vec<_>>>()?;
        if offsets[0] != 0 || offsets[m] != dim || offsets.windows(2).any(|v| v[0] >= v[1]) {
            return Err(bad());
        }
        let mean = read_floats(r, dim, storage)?;
        let centers = read_floats(r, dim * 256, storage)?;
        let table = FixedChunkPQTable {
            centers,
            mean,
            offsets: orion_ann_platform::region::ArrayView::copy_from(&offsets, storage)?,
        };
        Self::from_parts(table, Codes::read_with_options(r, rows * m, storage)?, rows)
    }
}

// Large reads keep both the runtime SHA-256 stream and the filesystem writer
// efficient; never funnel the codebook through one read/write call per scalar.
fn read_floats(
    reader: &mut dyn Read,
    count: usize,
    storage: &orion_ann_platform::region::RegionOptions,
) -> Result<orion_ann_platform::region::ArrayView<f32>> {
    orion_ann_platform::region::ArrayView::<f32>::build_rows::<Error>(
        count,
        1,
        64,
        storage,
        |_, values| {
            reader.read_exact(bytemuck::cast_slice_mut(values))?;
            #[cfg(target_endian = "big")]
            for value in values.iter_mut() {
                *value = f32::from_bits(u32::from_le(value.to_bits()));
            }
            if values.iter().any(|value| !value.is_finite()) {
                return Err(Error::cache("PQ codebook contains non-finite coordinates"));
            }
            Ok(())
        },
    )
}
fn write_floats(writer: &mut dyn Write, values: &[f32]) -> Result<()> {
    #[cfg(target_endian = "little")]
    writer.write_all(bytemuck::cast_slice(values))?;
    #[cfg(target_endian = "big")]
    {
        let mut bytes = Vec::with_capacity(4096);
        for block in values.chunks(1024) {
            bytes.clear();
            for value in block {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            writer.write_all(&bytes)?;
        }
    }
    Ok(())
}
