/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! # Layout at a glance
//!
//! ```text
//! memory: [norm_squared(row 0) ... norm_squared(row N-1)] as u32 or u64
//! file:   ["U8NS": 4 B][version: u32][N: u64][D: u64][N squared norms]
//! ```
//!
//! All file fields are little-endian. The dimension determines the norm word
//! width: use u32 if `D * 255^2` fits, otherwise u64. This is an auxiliary array,
//! not a duplicate vector dataset; the original rows remain in `PaddedDataset<u8>`.
//! Owned and private-mapped arrays use the same per-row indexing.

//! Exact squared norms for native bytes. Independent of quantization and graph topology.
use crate::{Error, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use orion_ann_diskann::model::PaddedDataset;
use orion_ann_platform::region::{ArrayView, Placement, RegionOptions};
use std::io::{Read, Write};
pub const NORM_CODEC: &str = "u8-norms";
pub(crate) const U32_NORM_DIMENSION: usize = (u32::MAX as u64 / 65025) as usize;
#[derive(Debug)]
enum Values {
    U32(ArrayView<u32>),
    U64(ArrayView<u64>),
}
#[derive(Debug)]
pub struct NativeU8Norms {
    values: Values,
    dimension: usize,
    rows: usize,
}
impl NativeU8Norms {
    /// Payload bytes for one exact squared norm per row. Width is determined
    /// from the worst case 255² × logical dimension, never a sampled maximum.
    ///
    /// # Errors
    /// Rejects zero dimensions, overflowing u64 norms and address-space overflow.
    pub fn estimated_bytes(rows: usize, dimension: usize) -> Result<usize> {
        if dimension == 0 || dimension as u128 * 65025 > u64::MAX as u128 {
            return Err(Error::invalid_argument(
                "native u8 norm dimension overflows u64",
            ));
        }
        rows.checked_mul(if dimension <= U32_NORM_DIMENSION {
            4
        } else {
            8
        })
        .ok_or_else(|| Error::invalid_argument("native u8 norms size overflow"))
    }
    pub(crate) fn build(base: &PaddedDataset<u8>, storage: &RegionOptions) -> Result<Self> {
        let dimension = base.layout().logical_dim();
        let rows = base.len();
        Self::estimated_bytes(rows, dimension)?;
        let norm = |i| {
            let sum = base.row(i).unwrap()[..dimension]
                .iter()
                .map(|&v| u64::from(v) * u64::from(v))
                .sum::<u64>();
            if sum == 0 {
                Err(
                    Error::invalid_argument("cosine requires nonzero base vectors")
                        .with_detail("row", i),
                )
            } else {
                Ok(sum)
            }
        };
        let values = if dimension <= U32_NORM_DIMENSION {
            Values::U32(ArrayView::build_rows(
                rows,
                1,
                64,
                storage,
                |start, out| {
                    for (i, v) in out.iter_mut().enumerate() {
                        *v = norm(start + i)? as u32;
                    }
                    Ok::<_, Error>(())
                },
            )?)
        } else {
            Values::U64(ArrayView::build_rows(
                rows,
                1,
                64,
                storage,
                |start, out| {
                    for (i, v) in out.iter_mut().enumerate() {
                        *v = norm(start + i)?;
                    }
                    Ok::<_, Error>(())
                },
            )?)
        };
        Ok(Self {
            values,
            dimension,
            rows,
        })
    }
    pub(crate) fn element_bytes(&self) -> usize {
        match &self.values {
            Values::U32(_) => 4,
            Values::U64(_) => 8,
        }
    }
    pub(crate) fn get(&self, id: usize) -> u64 {
        match &self.values {
            Values::U32(v) => v.as_slice()[id] as u64,
            Values::U64(v) => v.as_slice()[id],
        }
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        match &self.values {
            Values::U32(v) => bytemuck::cast_slice(v.as_slice()),
            Values::U64(v) => bytemuck::cast_slice(v.as_slice()),
        }
    }
    pub(crate) fn placement(&self) -> Placement {
        match &self.values {
            Values::U32(v) => v.placement(),
            Values::U64(v) => v.placement(),
        }
    }
    pub(crate) fn encode(&self, w: &mut dyn Write) -> Result<()> {
        w.write_all(b"U8NS")?;
        w.write_u32::<LittleEndian>(1)?;
        w.write_u64::<LittleEndian>(self.rows as u64)?;
        w.write_u64::<LittleEndian>(self.dimension as u64)?;
        match &self.values {
            Values::U32(v) => {
                for &x in v.as_slice() {
                    w.write_u32::<LittleEndian>(x)?;
                }
            }
            Values::U64(v) => {
                for &x in v.as_slice() {
                    w.write_u64::<LittleEndian>(x)?;
                }
            }
        }
        Ok(())
    }
    pub(crate) fn decode(
        r: &mut dyn Read,
        bytes: u64,
        rows: usize,
        dimension: usize,
        storage: &RegionOptions,
    ) -> Result<Self> {
        let mut magic = [0; 4];
        r.read_exact(&mut magic)?;
        if &magic != b"U8NS"
            || r.read_u32::<LittleEndian>()? != 1
            || r.read_u64::<LittleEndian>()? != rows as u64
            || r.read_u64::<LittleEndian>()? != dimension as u64
            || bytes != 24 + Self::estimated_bytes(rows, dimension)? as u64
        {
            return Err(Error::cache(
                "native u8 norm shape, version or length mismatch",
            ));
        }
        let limit = dimension as u64 * 65025;
        let values = if dimension <= U32_NORM_DIMENSION {
            Values::U32(ArrayView::build_rows(rows, 1, 64, storage, |_, out| {
                r.read_u32_into::<LittleEndian>(out)?;
                if out.iter().any(|&v| v == 0 || v as u64 > limit) {
                    return Err(Error::cache("invalid u8 squared norm"));
                }
                Ok::<_, Error>(())
            })?)
        } else {
            Values::U64(ArrayView::build_rows(rows, 1, 64, storage, |_, out| {
                r.read_u64_into::<LittleEndian>(out)?;
                if out.iter().any(|&v| v == 0 || v > limit) {
                    return Err(Error::cache("invalid u8 squared norm"));
                }
                Ok::<_, Error>(())
            })?)
        };
        let mut tail = [0];
        if r.read(&mut tail)? != 0 {
            return Err(Error::cache("trailing u8 norm bytes"));
        }
        Ok(Self {
            values,
            rows,
            dimension,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use orion_ann_vector::{IpU8Distance, PaddedLayout};
    #[test]
    fn width_budget_codec_and_mapping_agree() {
        for dimension in [1, 128, U32_NORM_DIMENSION, U32_NORM_DIMENSION + 1] {
            let base = PaddedDataset::from_packed(
                &vec![255u8; dimension * 2],
                PaddedLayout::for_kernel::<IpU8Distance>(dimension).unwrap(),
            )
            .unwrap();
            for placement in [Placement::Resident, Placement::Mapped] {
                let options = RegionOptions {
                    placement,
                    directory: None,
                };
                let norms = NativeU8Norms::build(&base, &options).unwrap();
                assert_eq!(
                    norms.bytes().len(),
                    NativeU8Norms::estimated_bytes(2, dimension).unwrap()
                );
                assert_eq!(norms.placement(), placement);
                assert_eq!(norms.get(1), dimension as u64 * 65025);
                let mut encoded = Vec::new();
                norms.encode(&mut encoded).unwrap();
                let restored = NativeU8Norms::decode(
                    &mut &encoded[..],
                    encoded.len() as u64,
                    2,
                    dimension,
                    &options,
                )
                .unwrap();
                drop(norms);

                assert_eq!(restored.get(1), dimension as u64 * 65025);
                for cut in [0, 23, encoded.len() - 1] {
                    assert!(
                        NativeU8Norms::decode(
                            &mut &encoded[..cut],
                            cut as u64,
                            2,
                            dimension,
                            &options
                        )
                        .is_err()
                    );
                }
                encoded[24..].fill(0);
                assert!(
                    NativeU8Norms::decode(
                        &mut &encoded[..],
                        encoded.len() as u64,
                        2,
                        dimension,
                        &options
                    )
                    .is_err()
                );
            }
        }
        assert!(NativeU8Norms::estimated_bytes(usize::MAX, 128).is_err());
        let base = PaddedDataset::from_packed(
            &[0u8; 32],
            PaddedLayout::for_kernel::<IpU8Distance>(32).unwrap(),
        )
        .unwrap();
        assert!(NativeU8Norms::build(&base, &RegionOptions::default()).is_err());
    }
}
