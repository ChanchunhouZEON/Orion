/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Little-endian STAG v4: magic, version, bounded JSON descriptor, then v3 rows.
use super::{NodePartitions, PartitionSource, PartitionSpec, validate_row};
use crate::{Error, Result};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

const MAGIC: u32 = 0x53544147;
const MAX_METADATA: usize = 16 * 1024;

/// Streaming staged input. Only one row is retained; no intermediate graph.
#[derive(Debug)]
pub struct StagedReader<R> {
    reader: R,
    spec: PartitionSpec,
    node: u32,
    remaining: u64,
    row: Vec<u32>,
}

impl<R: Read> StagedReader<R> {
    /// Open a v4 stream and validate its bounded descriptor before allocating rows.
    /// `byte_len` is the exact stream length; EOF and trailing bytes are checked.
    pub fn new(mut reader: R, byte_len: u64) -> Result<Self> {
        if byte_len < 12
            || reader.read_u32::<LittleEndian>()? != MAGIC
            || reader.read_u32::<LittleEndian>()? != 4
        {
            return Err(Error::invalid_argument("expected supplier STAG v4 header"));
        }
        let len = reader.read_u32::<LittleEndian>()? as usize;
        if len == 0 || len > MAX_METADATA || len as u64 > byte_len - 12 {
            return Err(Error::invalid_argument("invalid staged descriptor length"));
        }
        let mut bytes = vec![0; len];
        reader.read_exact(&mut bytes)?;
        let spec: PartitionSpec = serde_json::from_slice(&bytes)?;
        Self::open_rows(reader, spec, byte_len - 12 - len as u64)
    }

    /// Explicit v3 adapter: callers supply the missing identity/metric metadata.
    /// The file's node count, capacities and entry must match the supplied spec;
    /// no entry replacement or ParlayANN-specific transformation is performed.
    pub fn legacy_v3(mut reader: R, byte_len: u64, spec: PartitionSpec) -> Result<Self> {
        if byte_len < 24
            || reader.read_u32::<LittleEndian>()? != MAGIC
            || reader.read_u32::<LittleEndian>()? != 3
        {
            return Err(Error::invalid_argument("expected legacy STAG v3 header"));
        }
        let fields = [spec.points, spec.max_degree, spec.max_extra, spec.entry];
        for expected in fields {
            if reader.read_u32::<LittleEndian>()? != expected {
                return Err(Error::invalid_argument(
                    "legacy staged header differs from supplied spec",
                ));
            }
        }
        Self::open_rows(reader, spec, byte_len - 24)
    }

    fn open_rows(reader: R, spec: PartitionSpec, remaining: u64) -> Result<Self> {
        spec.validate()?;
        if remaining < spec.points as u64 * 12 {
            return Err(Error::invalid_argument("truncated staged row headers"));
        }
        Ok(Self {
            reader,
            spec,
            node: 0,
            remaining,
            row: Vec::new(),
        })
    }
    pub fn spec(&self) -> &PartitionSpec {
        &self.spec
    }
}

impl<R: Read> PartitionSource for StagedReader<R> {
    fn next_partition(&mut self) -> Result<Option<NodePartitions<'_>>> {
        if self.node == self.spec.points {
            let mut tail = [0];
            if self.remaining != 0 || self.reader.read(&mut tail)? != 0 {
                return Err(Error::invalid_argument("trailing staged bytes"));
            }
            return Ok(None);
        }
        if self.remaining < 12 {
            return Err(Error::invalid_argument("truncated staged row"));
        }
        let local = self.reader.read_u32::<LittleEndian>()? as usize;
        let remote = self.reader.read_u32::<LittleEndian>()? as usize;
        let extra = self.reader.read_u32::<LittleEndian>()? as usize;
        let navigation = local
            .checked_add(remote)
            .ok_or_else(|| Error::invalid_argument("partition count overflow"))?;
        let total = navigation
            .checked_add(extra)
            .ok_or_else(|| Error::invalid_argument("partition count overflow"))?;
        if navigation > self.spec.max_degree as usize
            || extra > self.spec.max_extra as usize
            || total as u64 >= self.spec.points as u64
            || total as u64 * 4 > self.remaining - 12
        {
            return Err(Error::invalid_argument("invalid staged row lengths")
                .with_detail("node", self.node));
        }
        self.row.clear();
        self.row
            .try_reserve(total)
            .map_err(|e| Error::invalid_argument(format!("staged row allocation failed: {e}")))?;
        self.row.resize(total, 0);
        self.reader.read_u32_into::<LittleEndian>(&mut self.row)?;
        self.remaining -= 12 + total as u64 * 4;
        self.node += 1;
        Ok(Some(NodePartitions {
            local: &self.row[..local],
            remote: &self.row[local..navigation],
            extra: &self.row[navigation..],
        }))
    }
}

/// Validate and encode v4 without changing partition order. Caller owns buffering,
/// flush and atomic publication; a failed stream must not be published.
pub fn write_staged(
    mut writer: impl Write,
    spec: &PartitionSpec,
    source: &mut (impl PartitionSource + ?Sized),
) -> Result<()> {
    spec.validate()?;
    let metadata = serde_json::to_vec(spec)?;
    if metadata.len() > MAX_METADATA {
        return Err(Error::invalid_argument("staged descriptor too large"));
    }
    writer.write_u32::<LittleEndian>(MAGIC)?;
    writer.write_u32::<LittleEndian>(4)?;
    writer.write_u32::<LittleEndian>(metadata.len() as u32)?;
    writer.write_all(&metadata)?;
    let mut scratch = Vec::new();
    for node in 0..spec.points {
        let row = source.next_partition()?.ok_or_else(|| {
            Error::invalid_argument("missing partition row").with_detail("node", node)
        })?;
        validate_row(spec, node, row, &mut scratch)?;
        for count in [row.local.len(), row.remote.len(), row.extra.len()] {
            writer.write_u32::<LittleEndian>(count as u32)?;
        }
        for &id in row.local.iter().chain(row.remote).chain(row.extra) {
            writer.write_u32::<LittleEndian>(id)?;
        }
    }
    if source.next_partition()?.is_some() {
        return Err(Error::invalid_argument("extra partition rows"));
    }
    Ok(())
}
