/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Sequential vector readers over `Read`.
//!
//! Files, stdin, pipes, and network streams share the same decoding path;
//! seeking and rewinding are never required.
//!
//! Query batches are exposed as flat row-major `f32` buffers. Byte-valued
//! inputs are widened only into this bounded query buffer and never into the
//! resident base representation.

use super::data::{validate_and_normalize_vectors, VectorFormat};
use crate::cascade::SearchMetric;
use std::io::Read;

pub trait QuerySource {
    fn dimension(&self) -> usize;
    fn count_hint(&self) -> Option<usize>;

    /// Replaces `output` with at most `max_queries` vectors.
    ///
    /// Returns the number of decoded vectors. Zero indicates EOF.
    fn read_batch(&mut self, output: &mut Vec<f32>, max_queries: usize) -> Result<usize, String>;
}

/// Reads one little-endian `u32`.
///
/// EOF before any byte is read returns `Ok(None)`. EOF after a partial word is
/// reported as a truncated header.
fn read_u32_le(reader: &mut impl Read) -> Result<Option<u32>, String> {
    let mut bytes = [0u8; 4];

    loop {
        match reader.read(&mut bytes[..1]) {
            Ok(0) => return Ok(None),
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {
                continue;
            }
            Err(error) => return Err(error.to_string()),
        }
    }

    reader
        .read_exact(&mut bytes[1..])
        .map_err(|error| format!("truncated record header: {error}"))?;

    Ok(Some(u32::from_le_bytes(bytes)))
}

fn is_binary_container(format: VectorFormat) -> bool {
    matches!(format, VectorFormat::Fbin | VectorFormat::U8bin)
}

fn is_byte_vector(format: VectorFormat) -> bool {
    matches!(format, VectorFormat::Bvecs | VectorFormat::U8bin)
}

fn bytes_per_element(format: VectorFormat) -> usize {
    if is_byte_vector(format) {
        1
    } else {
        std::mem::size_of::<f32>()
    }
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
    ) -> Result<Self, String> {
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
            .checked_mul(bytes_per_element(format))
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
    ) -> Result<(Option<usize>, Option<u32>), String> {
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
    ) -> Result<(Option<usize>, Option<u32>), String> {
        Self::validate_dimension(first_dimension, expected_dimension)?;

        Ok((None, Some(first_dimension)))
    }

    fn validate_dimension(actual: u32, expected: usize) -> Result<(), String> {
        if actual as usize != expected {
            return Err(format!(
                "query dimension {actual} != configured dimension {expected}"
            ));
        }

        Ok(())
    }

    /// Advances to the next row and validates its record header.
    ///
    /// Returns `false` at clean EOF.
    fn begin_next_row(&mut self) -> Result<bool, String> {
        if let Some(declared_count) = self.declared_count {
            return self.begin_binary_row(declared_count);
        }

        self.begin_record_stream_row()
    }

    fn begin_binary_row(&mut self, declared_count: usize) -> Result<bool, String> {
        if self.rows_read < declared_count {
            return Ok(true);
        }

        if read_u32_le(&mut self.reader)?.is_some() {
            return Err("trailing bytes after binary queries".into());
        }

        Ok(false)
    }

    fn begin_record_stream_row(&mut self) -> Result<bool, String> {
        let dimension = match self.first_row_dimension.take() {
            Some(dimension) => Some(dimension),
            None => read_u32_le(&mut self.reader)?,
        };

        let Some(dimension) = dimension else {
            return Ok(false);
        };

        if dimension as usize != self.dimension {
            return Err(format!(
                "query {} dimension {dimension} != {}",
                self.rows_read, self.dimension,
            ));
        }

        Ok(true)
    }

    fn read_payload(&mut self) -> Result<(), String> {
        self.reader
            .read_exact(&mut self.record_buf)
            .map_err(|error| format!("query {} payload is truncated: {error}", self.rows_read,))
    }

    fn decode_payload(&self, output: &mut Vec<f32>) {
        if is_byte_vector(self.format) {
            output.extend(self.record_buf.iter().map(|&value| f32::from(value)));
        } else {
            output.extend(
                self.record_buf
                    .chunks_exact(std::mem::size_of::<f32>())
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())),
            );
        }
    }
}

impl<R: Read> QuerySource for VectorReader<R> {
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn count_hint(&self) -> Option<usize> {
        self.declared_count
    }

    fn read_batch(&mut self, output: &mut Vec<f32>, max_queries: usize) -> Result<usize, String> {
        if max_queries == 0 {
            return Err("query batch size must be positive".into());
        }

        output.clear();

        for _ in 0..max_queries {
            if !self.begin_next_row()? {
                break;
            }

            self.read_payload()?;
            self.decode_payload(output);

            self.rows_read += 1;
        }

        validate_and_normalize_vectors(output, self.dimension, self.metric)?;

        Ok(output.len() / self.dimension)
    }
}

/// Sequential ground-truth evaluator input.
///
/// Ground truth is optional for search. When supplied, however, every row must
/// align with the query stream and all rows must use a consistent width.
pub struct GroundTruthReader<R> {
    reader: R,
    k: usize,
    base_count: usize,
    row_width: Option<usize>,
    rows_read: usize,
}

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

    pub fn next(&mut self) -> Result<Option<Vec<u32>>, String> {
        let Some(width) = read_u32_le(&mut self.reader)? else {
            return Ok(None);
        };

        let width = width as usize;
        self.validate_width(width)?;

        let mut ids = Vec::with_capacity(self.k);

        for rank in 0..width {
            let id = read_u32_le(&mut self.reader)?.ok_or("truncated ground truth row")?;

            if id as usize >= self.base_count {
                return Err(format!(
                    "ground truth ID {id} is outside loaded base count {}",
                    self.base_count,
                ));
            }

            if rank < self.k {
                ids.push(id);
            }
        }

        self.row_width = Some(width);
        self.rows_read += 1;

        Ok(Some(ids))
    }

    fn validate_width(&self, width: usize) -> Result<(), String> {
        let inconsistent = self.row_width.is_some_and(|previous| previous != width);

        if width < self.k || width > self.base_count || inconsistent {
            return Err(format!(
                "ground truth row {} has invalid width {width}; \
                 require consistent width >= k={} and <= base count {}",
                self.rows_read, self.k, self.base_count,
            ));
        }

        Ok(())
    }
}

/// Opens the configured query stream.
///
/// `-` reads from stdin. Any other path is opened as a regular file. Neither
/// source requires `Seek`.
pub fn open_queries(
    config: &super::config::ResolvedRunConfig,
) -> Result<Box<dyn QuerySource>, String> {
    use std::io::BufReader;

    let path = config
        .query
        .as_deref()
        .ok_or("--query is required for search")?;

    let reader: Box<dyn Read> = if path == std::path::Path::new("-") {
        Box::new(std::io::stdin())
    } else {
        Box::new(
            std::fs::File::open(path).map_err(|error| format!("{}: {error}", path.display(),))?,
        )
    };

    let reader = VectorReader::new(
        BufReader::new(reader),
        config.query_format,
        config.dimension,
        config.metric,
    )?;

    Ok(Box::new(reader))
}
