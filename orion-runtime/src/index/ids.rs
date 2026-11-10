/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Optional application identity, independent of graph topology and cache keys.
use crate::{Error, ErrorCode, Result};
use std::io::Write;

pub(super) struct IdMap(pub Vec<u64>);

impl IdMap {
    pub fn validate(ids: Vec<u64>, points: usize, budget: Option<f64>) -> Result<Self> {
        if ids.len() != points {
            return Err(Error::invalid_argument(
                "external ID count must equal indexed point count",
            ));
        }
        reserve_budget(points, budget, 1)?;
        // Sorting a temporary copy avoids a permanent reverse map and its hash
        // table overhead. Its peak is included in the preparation budget.
        let mut sorted = Vec::new();
        sorted
            .try_reserve_exact(ids.len())
            .map_err(|e| Error::new(ErrorCode::ResourceExhausted, e.to_string()))?;
        sorted.extend_from_slice(&ids);
        sorted.sort_unstable();
        if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(Error::invalid_argument("external IDs must be unique"));
        }
        Ok(Self(ids))
    }

    pub fn save(&self, store: &std::path::Path) -> Result<String> {
        crate::storage::put(store, "ids-u64-le", |writer| {
            if cfg!(target_endian = "little") {
                for block in bytemuck::cast_slice::<u64, u8>(&self.0)
                    .chunks(orion_ann_platform::buffered_io::FILE_WRITE_BYTES)
                {
                    writer.write_all(block)?;
                }
            } else {
                for &id in &self.0 {
                    writer.write_all(&id.to_le_bytes())?;
                }
            }
            Ok(())
        })
        .map(|(_, hash)| hash)
    }

    pub fn load(
        store: &std::path::Path,
        digest: &str,
        points: usize,
        budget: Option<f64>,
    ) -> Result<Self> {
        reserve_budget(points, budget, 2)?;
        let path = crate::storage::object(store, digest, "ids-u64-le")?;
        let ids = crate::integrity::read_verified(
            &path,
            "external IDs",
            digest,
            Some(points as u64 * 8),
            |reader, _| {
                let mut ids = Vec::new();
                ids.try_reserve_exact(points)
                    .map_err(|e| Error::new(ErrorCode::ResourceExhausted, e.to_string()))?;
                ids.resize(points, 0u64);
                for block in bytemuck::cast_slice_mut::<u64, u8>(&mut ids)
                    .chunks_mut(orion_ann_platform::buffered_io::FILE_WRITE_BYTES)
                {
                    reader.read_exact(block)?;
                }
                for id in &mut ids {
                    *id = u64::from_le(*id);
                }
                Ok(ids)
            },
        )?
        .value;
        Self::validate(ids, points, budget)
    }
}

fn reserve_budget(points: usize, budget: Option<f64>, pending_buffers: u64) -> Result<()> {
    crate::config::resources::validate_memory_budget(budget)?;
    if points == 0 || points >= u32::MAX as usize {
        return Err(Error::invalid_argument(
            "external ID count is outside index bounds",
        ));
    }
    let bytes = (points as u64) * 16;
    // A caller-supplied mapping is already charged to observed memory. On
    // load both buffers are pending; during validation only the copy is pending.
    let pending = points as u64 * 8 * pending_buffers;
    let available = orion_ann_platform::resources::MemoryResources::detect().available_bytes;
    if budget.is_some_and(|gib| gib * 1073741824.0 <= bytes as f64)
        || available.is_some_and(|free| pending > free)
    {
        return Err(Error::new(
            ErrorCode::ResourceExhausted,
            "external ID validation requires the mapping plus a sorting copy",
        )
        .with_detail("external_ids_resident_bytes", points as u64 * 8)
        .with_detail("external_ids_validation_peak_bytes", bytes));
    }
    Ok(())
}
