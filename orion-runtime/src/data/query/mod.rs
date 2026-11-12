/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Sequential vector readers over `Read`.
//!
//! Files, stdin, pipes, and network streams share the same decoding path;
//! seeking and rewinding are never required.
//!
//! Typed batches preserve file coordinates. Only legacy benchmark readers
//! request f32 transport; production preparation is per query and base agnostic.

use crate::config::cascade::SearchMetric;
use crate::data::{validate_and_normalize_vectors, VectorFormat};
use crate::{Error, QueryBuffer, Result};
mod encoding;
use std::io::Read;

pub trait QuerySource {
    #[cfg(any(test, feature = "cli", feature = "benchmark"))]
    fn dimension(&self) -> usize;
    #[cfg(any(test, feature = "cli", feature = "benchmark"))]
    fn count_hint(&self) -> Option<usize>;

    /// Replaces `output` with at most `max_queries` vectors.
    ///
    /// Returns the number of decoded vectors. Zero indicates EOF.
    fn read_batch(&mut self, output: &mut Vec<f32>, max_queries: usize) -> Result<usize>;

    /// Native typed transport. Legacy custom f32 sources use this default;
    /// built-in file readers override it without conversion or normalization.
    fn read_typed_batch(&mut self, output: &mut QueryBuffer, max_queries: usize) -> Result<usize> {
        let mut values = Vec::new();
        let count = self.read_batch(&mut values, max_queries)?;
        *output = QueryBuffer::F32(values);
        Ok(count)
    }
}

/// Reads one little-endian `u32`.
///
/// EOF before any byte is read returns `Ok(None)`. EOF after a partial word is
/// reported as a truncated header.
fn read_u32_le(reader: &mut impl Read) -> Result<Option<u32>> {
    let mut bytes = [0u8; 4];

    loop {
        match reader.read(&mut bytes[..1]) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                continue;
            }
            Err(error) => return Err(error.into()),
        }
    }

    reader.read_exact(&mut bytes[1..]).map_err(|error| {
        let message = format!("truncated record header: {error}");
        Error::from(error).with_message(message)
    })?;

    Ok(Some(u32::from_le_bytes(bytes)))
}

fn is_binary_container(format: VectorFormat) -> bool {
    format.has_file_header()
}

pub struct VectorReader<R> {
    reader: R,
    format: VectorFormat,
    metric: SearchMetric,
    dimension: usize,

    /// Number of rows declared by a binary container header.
    ///
    /// `None` for record-oriented formats such as `fvecs` and `bvecs`.
    declared_count: Option<usize>,

    rows_read: usize,

    /// `fvecs` / `bvecs` place the dimension at the beginning of every row.
    ///
    /// Construction has already consumed the first row's dimension in order
    /// to validate the stream, so it is retained here for the first read.
    first_row_dimension: Option<u32>,

    /// Reusable buffer for one encoded vector payload.
    record_buf: Vec<u8>,
}

impl<R: Read> VectorReader<R> {
    pub fn new(
        mut reader: R,
        format: VectorFormat,
        dimension: usize,
        metric: SearchMetric,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err("query dimension must be positive".into());
        }

        let first_word = read_u32_le(&mut reader)?.ok_or("empty query source")?;

        let (declared_count, first_row_dimension) = if is_binary_container(format) {
            Self::read_binary_header(&mut reader, first_word, dimension)?
        } else {
            Self::read_record_stream_header(first_word, dimension)?
        };

        let record_bytes = dimension
            .checked_mul(format.coordinate_bytes())
            .ok_or("query dimension overflow")?;

        Ok(Self {
            reader,
            format,
            metric,
            dimension,
            declared_count,
            rows_read: 0,
            first_row_dimension,
            record_buf: vec![0; record_bytes],
        })
    }

    fn read_binary_header(
        reader: &mut R,
        declared_count: u32,
        expected_dimension: usize,
    ) -> Result<(Option<usize>, Option<u32>)> {
        if declared_count == 0 {
            return Err("empty query source".into());
        }

        let actual_dimension = read_u32_le(reader)?.ok_or("missing binary query dimension")?;

        Self::validate_dimension(actual_dimension, expected_dimension)?;

        Ok((Some(declared_count as usize), None))
    }

    fn read_record_stream_header(
        first_dimension: u32,
        expected_dimension: usize,
    ) -> Result<(Option<usize>, Option<u32>)> {
        Self::validate_dimension(first_dimension, expected_dimension)?;

        Ok((None, Some(first_dimension)))
    }

    fn validate_dimension(actual: u32, expected: usize) -> Result<()> {
        if actual as usize != expected {
            return Err(
                Error::dimension("queries.dimension", expected, actual as usize).at("read_queries"),
            );
        }

        Ok(())
    }

    /// Advances to the next row and validates its record header.
    ///
    /// Returns `false` at clean EOF.
    fn begin_next_row(&mut self) -> Result<bool> {
        if let Some(declared_count) = self.declared_count {
            return self.begin_binary_row(declared_count);
        }

        self.begin_record_stream_row()
    }

    fn begin_binary_row(&mut self, declared_count: usize) -> Result<bool> {
        if self.rows_read < declared_count {
            return Ok(true);
        }

        if read_u32_le(&mut self.reader)?.is_some() {
            return Err("trailing bytes after binary queries".into());
        }

        Ok(false)
    }

    fn begin_record_stream_row(&mut self) -> Result<bool> {
        let dimension = match self.first_row_dimension.take() {
            Some(dimension) => Some(dimension),
            None => read_u32_le(&mut self.reader)?,
        };

        let Some(dimension) = dimension else {
            return Ok(false);
        };

        if dimension as usize != self.dimension {
            return Err(Error::from(format!(
                "query {} dimension {dimension} != {}",
                self.rows_read, self.dimension,
            )));
        }

        Ok(true)
    }

    fn read_payload(&mut self) -> Result<()> {
        self.reader
            .read_exact(&mut self.record_buf)
            .map_err(|error| {
                Error::invalid_data(format!(
                    "query {} payload is truncated: {error}",
                    self.rows_read
                ))
                .at("read_queries")
                .with_detail("row", self.rows_read)
                .with_source(error)
            })
    }
}

impl<R: Read> QuerySource for VectorReader<R> {
    #[cfg(any(test, feature = "cli", feature = "benchmark"))]
    fn dimension(&self) -> usize {
        self.dimension
    }

    #[cfg(any(test, feature = "cli", feature = "benchmark"))]
    fn count_hint(&self) -> Option<usize> {
        self.declared_count
    }

    fn read_batch(&mut self, output: &mut Vec<f32>, max_queries: usize) -> Result<usize> {
        let mut typed = QueryBuffer::default();
        let count = self.read_typed_batch(&mut typed, max_queries)?;
        *output = encoding::legacy_f32(&typed)?;
        validate_and_normalize_vectors(output, self.dimension, self.metric)?;
        Ok(count)
    }

    fn read_typed_batch(&mut self, output: &mut QueryBuffer, max_queries: usize) -> Result<usize> {
        if max_queries == 0 {
            return Err("query batch size must be positive".into());
        }
        encoding::reset(output, self.format);
        for _ in 0..max_queries {
            if !self.begin_next_row()? {
                break;
            }
            self.read_payload()?;
            encoding::append(output, &self.record_buf);
            self.rows_read += 1;
        }
        output.as_input().validate()?;
        Ok(output.len() / self.dimension)
    }
}

/// Sequential ground-truth evaluator input.
///
/// Ground truth is optional for search. When supplied, however, every row must
/// align with the query stream and all rows must use a consistent width.
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub struct GroundTruthReader<R> {
    reader: R,
    k: usize,
    base_count: usize,
    row_width: Option<usize>,
    rows_read: usize,
}

#[cfg(any(feature = "cli", feature = "benchmark"))]
impl<R: Read> GroundTruthReader<R> {
    pub fn new(reader: R, k: usize, base_count: usize) -> Self {
        Self {
            reader,
            k,
            base_count,
            row_width: None,
            rows_read: 0,
        }
    }

    /// Read one row, distinguishing a malformed row from a clean end of input.
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible reader API returns Result<Option<T>> and preserves the public contract"
    )]
    pub fn next(&mut self) -> Result<Option<Vec<u32>>> {
        let Some(width) = read_u32_le(&mut self.reader)? else {
            return Ok(None);
        };

        let width = width as usize;
        self.validate_width(width)?;

        let mut ids = Vec::with_capacity(self.k);

        for rank in 0..width {
            let id = read_u32_le(&mut self.reader)?.ok_or("truncated ground truth row")?;

            if id as usize >= self.base_count {
                return Err(Error::from(format!(
                    "ground truth ID {id} is outside loaded base count {}",
                    self.base_count,
                )));
            }

            if rank < self.k {
                ids.push(id);
            }
        }

        self.row_width = Some(width);
        self.rows_read += 1;

        Ok(Some(ids))
    }

    fn validate_width(&self, width: usize) -> Result<()> {
        let inconsistent = self.row_width.is_some_and(|previous| previous != width);

        if width < self.k || width > self.base_count || inconsistent {
            return Err(Error::from(format!(
                "ground truth row {} has invalid width {width}; \
                 require consistent width >= k={} and <= base count {}",
                self.rows_read, self.k, self.base_count,
            )));
        }

        Ok(())
    }
}

/// Opens the configured query stream.
///
/// `-` reads from stdin. Any other path is opened as a regular file. Neither
/// source requires `Seek`.
#[cfg(any(feature = "cli", feature = "benchmark"))]
pub fn open_queries(config: &crate::config::ResolvedRunConfig) -> Result<Box<dyn QuerySource>> {
    let path = config
        .query
        .as_deref()
        .ok_or("--query is required for search")?;
    // Native cosine keeps byte coordinates; Index validates zero vectors and
    // applies norm-adjusted distances. The decoder must not normalize them.
    let decode_metric = if config.storage() == crate::VectorStorageKind::U8 {
        SearchMetric::SquaredL2
    } else {
        config.metric
    };
    open_stream(path, config.query_format, config.dimension, decode_metric)
}

#[cfg(feature = "cli")]
pub fn open_snapshot_queries(
    path: &std::path::Path,
    format: VectorFormat,
    dimension: usize,
    storage: crate::VectorStorageKind,
) -> Result<Box<dyn QuerySource>> {
    let _ = storage; // Query format is independent of base storage.
    open_stream(path, format, dimension, SearchMetric::SquaredL2)
}

/// Decode packed query rows. Index normalizes the decoded values during search.
#[cfg(feature = "file-queries")]
pub fn read_query_file(
    path: &std::path::Path,
    format: Option<VectorFormat>,
    dimension: usize,
    _storage: crate::VectorStorageKind,
) -> Result<Vec<f32>> {
    encoding::legacy_f32(&read_query_buffer(path, format, dimension)?)
}

/// Read typed packed rows without narrowing. Binary integer formats carry the
/// same little-endian [count: u32, dimension: u32] header as fbin/u8bin.
#[cfg(feature = "file-queries")]
pub fn read_query_buffer(
    path: &std::path::Path,
    format: Option<VectorFormat>,
    dimension: usize,
) -> Result<QueryBuffer> {
    let format = format
        .map(Ok)
        .unwrap_or_else(|| VectorFormat::infer(path))?;
    let file = std::fs::File::open(path).map_err(|e| Error::io("open", path, e))?;
    let mut reader = VectorReader::new(
        std::io::BufReader::new(file),
        format,
        dimension,
        SearchMetric::SquaredL2,
    )?;
    let mut values = encoding::empty(format);
    let mut batch = QueryBuffer::default();
    while reader.read_typed_batch(&mut batch, 1024)? != 0 {
        values.extend(&batch)?;
    }
    Ok(values)
}

#[cfg(any(feature = "cli", feature = "benchmark"))]
fn open_stream(
    path: &std::path::Path,
    format: VectorFormat,
    dimension: usize,
    metric: SearchMetric,
) -> Result<Box<dyn QuerySource>> {
    use std::io::BufReader;
    let reader: Box<dyn Read> = if path == std::path::Path::new("-") {
        Box::new(std::io::stdin())
    } else {
        Box::new(std::fs::File::open(path).map_err(|error| {
            let message = format!("{}: {error}", path.display(),);
            Error::from(error).with_message(message)
        })?)
    };

    let reader = VectorReader::new(BufReader::new(reader), format, dimension, metric)?;

    Ok(Box::new(reader))
}

// Keep diagnostics bounded: do not format vector payloads or working buffers.
impl<R> std::fmt::Debug for VectorReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorReader")
            .field("format", &self.format)
            .field("metric", &self.metric)
            .field("dimension", &self.dimension)
            .field("declared_count", &self.declared_count)
            .field("rows_read", &self.rows_read)
            .finish_non_exhaustive()
    }
}

#[cfg(any(feature = "cli", feature = "benchmark"))]
impl<R> std::fmt::Debug for GroundTruthReader<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroundTruthReader")
            .field("k", &self.k)
            .field("rows_read", &self.rows_read)
            .field("base_count", &self.base_count)
            .finish_non_exhaustive()
    }
}
