/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Portable graph metadata and allocation shape.
use crate::{Error, PhasedGraph, Result};
use serde::{Deserialize, Serialize};

/// Display provenance, never a filename, namespace or executable backend name.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionOrigin {
    /// Supplier-defined algorithm name recorded as provenance.
    pub name: String,

    /// Optional supplier algorithm revision recorded with the graph.
    pub revision: Option<String>,
}

/// Coordinate and preprocessing contract of an imported graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PartitionMetric {
    SquaredL2,
    InnerProduct,
    /// Cosine geometry: normalized f32 rows or norm-adjusted native u8 rows.
    Cosine,
}

/// Scalar encoding declared by a supplier partition file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PartitionElement {
    F32,
    U8,
}

/// Portable graph description. Counts are logical, before SIMD padding.
/// Optional base SHA-256 binds the original input bytes; it is an integrity
/// assertion, not authentication of an untrusted supplier.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartitionSpec {
    /// Supplier identity recorded alongside the imported partitions.
    pub origin: PartitionOrigin,

    /// Number of base vectors identified by this record.
    pub points: u32,

    /// Logical coordinates per original vector, excluding padding.
    pub dimension: u32,

    /// Scalar encoding of the base expected by the supplied graph.
    pub element: PartitionElement,

    /// Original metric used to interpret coordinates and scores.
    pub metric: PartitionMetric,

    /// Internal vertex ID from which graph traversal starts.
    pub entry: u32,

    /// Maximum number of navigation neighbors retained per node.
    pub max_degree: u32,

    /// Maximum extra-candidate count stored per node.
    pub max_extra: u32,

    /// Content identity binding the graph or snapshot to its original base.
    pub base_sha256: Option<String>,
}

impl PartitionSpec {
    /// Validate bounded metadata and checked allocation arithmetic.
    pub fn validate(&self) -> Result<()> {
        let name = &self.origin.name;
        if name.trim().is_empty()
            || name.len() > 256
            || name.chars().any(char::is_control)
            || self
                .origin
                .revision
                .as_ref()
                .is_some_and(|s| s.len() > 256 || s.chars().any(char::is_control))
        {
            return Err(Error::invalid_argument(
                "supplier name/revision must be bounded printable text",
            ));
        }
        if self.points == 0
            || self.points == u32::MAX
            || self.dimension == 0
            || self.entry >= self.points
            || self.max_degree == 0
        {
            return Err(Error::invalid_argument(
                "invalid partition shape, entry or degree",
            ));
        }
        if self.base_sha256.as_ref().is_some_and(|s| {
            s.len() != 64
                || !s
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }) {
            return Err(Error::invalid_argument(
                "base_sha256 must be 64 lowercase hex digits",
            ));
        }
        self.slab_bytes()?;
        Ok(())
    }

    /// Final graph slab allocation (excluding its 16-byte serialized header).
    pub fn slab_bytes(&self) -> Result<usize> {
        PhasedGraph::checked_stride(
            self.points as usize,
            self.max_degree,
            self.max_extra as usize,
        )?
        .checked_mul(4)
        .and_then(|n| n.checked_mul(self.points as usize))
        .ok_or_else(|| Error::invalid_argument("partition allocation size overflow"))
    }
}
