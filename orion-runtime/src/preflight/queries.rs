/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Known replay/stream buffers, separate from persistent index placement.
use super::estimate::checked_mul_bytes;
use crate::{
    config::{ResolvedRunConfig, VectorStorageKind},
    Result,
};

/// Known query, result and evaluation buffers for a replay or streaming run.
pub(super) struct QueryBuffers {
    /// Maximum query rows retained by the selected execution mode.
    pub buffered: usize,

    /// Decoded and search-ready query payloads, including native-byte worker buffers when needed.
    pub query_bytes: u64,

    /// Internal top-k ID storage for buffered queries.
    pub result_bytes: u64,

    /// Ground-truth IDs retained for replay or the current streaming row.
    pub gt_bytes: u64,

    /// Whether replay requires a query count that has not yet been established.
    pub unknown_replay: bool,
}
impl QueryBuffers {
    /// Estimates execution buffers from query count and batching policy, rejecting byte-count overflow.
    ///
    /// Graph-only preparation has no query buffers; an unknown replay size remains
    /// explicit rather than being interpreted as a complete zero-byte estimate.
    pub(super) fn estimate(
        config: &ResolvedRunConfig,
        query_count: Option<usize>,
        layout: orion_ann_vector::PaddedLayout,
    ) -> Result<Self> {
        let storage = config.storage();
        // Report known working buffers separately from resident index storage.
        // Unknown lifetimes/implementations remain explicit rather than counted as zero.
        let unknown_replay = config.replay_queries && query_count.is_none() && !config.prepare_only;
        let buffered = if config.prepare_only {
            0
        } else if config.replay_queries {
            query_count.unwrap_or(0)
        } else {
            query_count
                .unwrap_or(
                    config
                        .query_batch_size
                        .max(config.sweep.calibration_samples),
                )
                .min(
                    config
                        .query_batch_size
                        .max(config.sweep.calibration_samples),
                )
        };
        let mut query_bytes = checked_mul_bytes(
            checked_mul_bytes(buffered as u64, config.dimension as u64)?,
            8,
        )?;
        if storage == VectorStorageKind::U8 {
            let native_bytes = checked_mul_bytes(
                buffered.min(config.sweep.threads) as u64,
                layout.stride_bytes() as u64,
            )?;
            query_bytes = query_bytes
                .checked_add(native_bytes)
                .ok_or("query buffers overflow")?;
        }
        let result_bytes = checked_mul_bytes(
            checked_mul_bytes(buffered as u64, config.sweep.k as u64)?,
            4,
        )?;
        let gt_bytes = if config.groundtruth.is_some() && !config.prepare_only {
            checked_mul_bytes(
                if config.replay_queries {
                    query_count.unwrap_or(0) as u64
                } else {
                    1
                },
                config.sweep.k as u64 * 4,
            )?
        } else {
            0
        };
        Ok(Self {
            buffered,
            query_bytes,
            result_bytes,
            gt_bytes,
            unknown_replay,
        })
    }
}
