/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Bounded-buffer vector loading into the allocation transferred to Orion.
use super::config::{ResolvedRunConfig, VectorStorageKind};
use crate::cascade::SearchMetric;
use diskann::common::AlignedBoxWithSlice;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const VECTOR_READ_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const SIMD_ALIGNMENT_BYTES: usize = 64;

/// Supported on-disk vector encodings.
///
/// The formats differ along two dimensions:
///
/// - **Record layout**
///   - [`VectorFormat::Fvecs`] and [`VectorFormat::Bvecs`] store a dimension
///     header before every vector.
///   - [`VectorFormat::Fbin`] and [`VectorFormat::U8bin`] store one global
///     `[count, dimension]` header followed by a contiguous row-major payload.
///
/// - **Coordinate encoding**
///   - [`VectorFormat::Fvecs`] and [`VectorFormat::Fbin`] store coordinates as
///     little-endian `f32`.
///   - [`VectorFormat::Bvecs`] and [`VectorFormat::U8bin`] store coordinates as
///     `u8`.
///
/// This distinction determines both shape inspection and how vectors are
/// decoded into the resident storage representation.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    serde::Serialize,
    clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum VectorFormat {
    Fvecs,
    Bvecs,
    Fbin,
    U8bin,
}

impl VectorFormat {
    /// Infers the vector encoding from the file extension.
    ///
    /// Paths without an extension _default_ to `fvecs` for backward
    /// compatibility. Unknown extensions require an explicit CLI format.
    pub fn infer(path: &Path) -> Result<Self, String> {
        match path.extension().and_then(|x| x.to_str()) {
            None | Some("fvecs") => Ok(Self::Fvecs),
            Some("bvecs") => Ok(Self::Bvecs),
            Some("fbin") => Ok(Self::Fbin),
            Some("u8bin") => Ok(Self::U8bin),
            _ => Err(format!(
                "unknown vector format for {}; specify --base-format/--query-format",
                path.display()
            )),
        }
    }

    /// Encoded bytes occupied by one coordinate in this format.
    fn coordinate_bytes(self) -> usize {
        if matches!(self, Self::Bvecs | Self::U8bin) {
            1
        } else {
            4
        }
    }

    /// Returns whether shape metadata is stored once at file scope.
    ///
    /// `fbin` and `u8bin` begin with `[count, dimension]`. In contrast,
    /// `fvecs` and `bvecs` encode the dimension separately for every row.
    fn has_file_header(self) -> bool {
        matches!(self, Self::Fbin | Self::U8bin)
    }
}

/// Shape metadata discovered without decoding the coordinate payload.
#[derive(Debug, serde::Serialize)]
pub struct VectorHeader {
    pub count: usize,
    pub dimension: usize,
}

/// Inspects vector shape and file size without decoding coordinate payloads.
///
/// For `fbin` / `u8bin`, shape is read directly from the file header and the
/// declared shape is checked against the total file length.
///
/// For `fvecs` / `bvecs`, the first record supplies the dimension and the row
/// count is inferred from the total file size. Dimensions on subsequent records
/// are intentionally validated later while decoding the requested prefix.
pub fn inspect_vector_file(path: &Path, format: VectorFormat) -> Result<VectorHeader, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;

    let file_bytes = file
        .metadata()
        .map_err(|e| e.to_string())?
        .len();

    let read_header_word =
        |file: &mut std::fs::File| -> Result<usize, String> {
            let mut word_bytes = [0; 4];
            file.read_exact(&mut word_bytes)
                .map_err(|e| e.to_string())?;

            Ok(u32::from_le_bytes(word_bytes) as usize)
        };

    // Binary formats have one [count, dimension] header; vecs formats repeat
    // a dimension word before every vector and derive their count from file size.
    let first_header_word = read_header_word(&mut file)?;
    let (count, dimension) = if format.has_file_header() {
        (
            first_header_word,
            read_header_word(&mut file)?
        )
    } else {
        // One vecs record consists of:
        //
        //     [dimension: u32][coordinate payload]
        //
        // All records are expected to have the same encoded size. Per-record
        // dimension values are checked later during actual decoding.
        let record_bytes = 4
            + first_header_word as u64 * format.coordinate_bytes() as u64;
        if file_bytes % record_bytes != 0 {
            return Err("truncated vector records".into());
        }
        (
            usize::try_from(file_bytes / record_bytes)
                .map_err(|e| e.to_string())?,
            first_header_word,
        )
    };

    if dimension == 0 || count == 0 {
        return Err("empty vectors".into());
    }

    // For file-header formats, verify that the declared shape exactly accounts
    // for the physical file size. This catches both truncated payloads and
    // unexpected trailing data before allocation begins.
    //
    // Vecs formats were already size-validated above, so their observed file
    // size is the expected size by construction.
    let expected_file_bytes = if format.has_file_header() {
        let payload_bytes = (count as u64)
            .checked_mul(dimension as u64)
            .and_then(|count| count.checked_mul(format.coordinate_bytes() as u64))
            .ok_or("file size overflow")?;

        8u64.checked_add(payload_bytes)
            .ok_or("file size overflow")?
    } else {
        file_bytes
    };

    if expected_file_bytes != file_bytes {
        return Err("binary header/file length mismatch".into());
    }

    Ok(VectorHeader { count, dimension })
}

/// Element abstraction used by the base-vector loader.
///
/// Decoding writes directly into the final resident representation rather than
/// first materializing an intermediate `Vec<f32>`. This is important for large
/// datasets because native-byte storage should not temporarily require an
/// additional full-precision copy.
///
/// Implementations define:
///
/// - how encoded byte coordinates are converted,
/// - whether encoded `f32` coordinates are accepted,
/// - what post-load normalization is required by the search metric.
pub trait BaseElement:
Default + Copy + Send + Sync + Into<f32>
{
    /// Resident storage representation selected by this element type.
    const STORAGE: VectorStorageKind;

    /// Converts one byte-encoded coordinate into resident storage.
    fn from_byte(value: u8) -> Self;

    /// Converts one floating-point coordinate into resident storage.
    ///
    /// Implementations may reject this conversion when doing so would require
    /// an implicit lossy transformation such as quantization.
    fn from_float(value: f32) -> Result<Self, String>;

    /// Performs representation-specific validation or normalization after load.
    fn normalize(
        data: &mut [Self],
        dimension: usize,
        metric: SearchMetric,
    ) -> Result<(), String>;
}

impl BaseElement for f32 {
    const STORAGE: VectorStorageKind = VectorStorageKind::F32;

    fn from_byte(value: u8) -> Self {
        f32::from(value)
    }

    fn from_float(value: f32) -> Result<Self, String> {
        Ok(value)
    }

    fn normalize(data: &mut [Self], dimension: usize, metric: SearchMetric) -> Result<(), String> {
        validate_and_normalize_vectors(data, dimension, metric)
    }
}

impl BaseElement for u8 {
    const STORAGE: VectorStorageKind = VectorStorageKind::U8;

    fn from_byte(value: u8) -> Self {
        value
    }

    fn from_float(_value: f32) -> Result<Self, String> {
        Err("native u8 loader does not quantize f32 input".into())
    }

    fn normalize(
        _data: &mut [Self],
        _dimension: usize,
        metric: SearchMetric,
    ) -> Result<(), String> {
        if metric != SearchMetric::L2 {
            return Err("native u8 requires L2".into());
        }
        Ok(())
    }
}

/// Decodes vectors directly into the caller-provided resident storage.
///
/// The length of `output` determines how many leading vectors are decoded.
/// This allows callers to load only a requested dataset prefix without scanning
/// or allocating the remainder of the file.
///
/// Only one encoded row is buffered at a time, so temporary decoding memory is
/// `O(dimension)` regardless of the number of vectors being loaded.
fn read_vectors_into<T: BaseElement>(
    path: &Path,
    format: VectorFormat,
    dimension: usize,
    output: &mut [T],
) -> Result<(), String> {
    let mut reader = BufReader::with_capacity(
        VECTOR_READ_BUFFER_BYTES,
        std::fs::File::open(path)
            .map_err(|e| e.to_string())?,
    );

    // File-header formats store `[count, dimension]` once at the beginning.
    // Shape validation has already been performed by `inspect_vector_file`, so
    // decoding can start directly at the coordinate payload.
    if format.has_file_header() {
        reader.seek(SeekFrom::Start(8)).map_err(|e| e.to_string())?;
    }

    // Reuse a single encoded-row buffer. The resident output is filled in place,
    // avoiding a second dataset-sized temporary allocation.
    let row_bytes = dimension
        .checked_mul(format.coordinate_bytes())
        .ok_or("row overflow")?;
    let mut encoded_row = vec![0; row_bytes];

    for values in output.chunks_exact_mut(dimension) {
        if !format.has_file_header() {
            // Vecs formats repeat the dimension before every record.
            //
            // Although `inspect_vector_file` used the first row to infer shape,
            // every decoded row must still be checked because a corrupted file
            // could contain a later record with a different dimension while
            // retaining a superficially valid total byte length.
            let mut dimension_header = [0; 4];

            reader
                .read_exact(&mut dimension_header)
                .map_err(|e| e.to_string())?;

            if u32::from_le_bytes(dimension_header) as usize != dimension {
                return Err("inconsistent per-record dimension".into());
            }
        }

        reader
            .read_exact(&mut encoded_row)
            .map_err(|e| e.to_string())?;

        if format.coordinate_bytes() == 1 {
            // Byte-valued formats can be written directly into either native
            // u8 storage or widened f32 storage without an intermediate row.
            for (coordinate, &encoded_coordinate) in
                values.iter_mut().zip(&encoded_row) {
                *coordinate = T::from_byte(encoded_coordinate);
            }
        } else {
            // Float-valued formats are decoded as little-endian f32 values.
            // Whether the resident representation accepts those values is a
            // policy of `BaseElement::from_float`.
            for (coordinate, encoded_coordinate) in
                values.iter_mut().zip(encoded_row.chunks_exact(4))
            {
                *coordinate =
                    T::from_float(f32::from_le_bytes(encoded_coordinate.try_into().unwrap()))?;
            }
        }
    }

    Ok(())
}

/// Temporary in-memory representation of a loaded benchmark dataset.
///
/// `LoadedDataset` owns the aligned allocation containing the base vectors,
/// independently of query and evaluation inputs. The base element type is `T`, which
/// defaults to `f32`.
///
/// # Ownership
///
/// The base vectors are stored in an [`AlignedBoxWithSlice<T>`] and are owned
/// by this struct until [`take_index_dataset`](Self::take_index_dataset) is
/// called. That method transfers the allocation directly into an
/// [`InmemDataset`](diskann::model::InmemDataset) without reallocating or
/// copying the base vectors.
///
/// After the transfer, [`base`](Self::base) becomes `None`.
///
/// # Workflow
///
/// A `LoadedBase` is normally constructed from a validated
/// [`ResolvedRunConfig`] using [`load`](Self::load). The resulting base
/// allocation can then be moved into an
/// [`Orion`](orion::index::compressed_index::Orion) index via
/// [`take_index_dataset`](Self::take_index_dataset).
///
/// This ownership-transfer path avoids keeping a second copy of the original
/// base dataset and eliminates the corresponding `memcpy`, which is especially
/// important for large datasets.
pub struct LoadedBase<T = f32> {
    /// Padded allocation, transferred to the index without copying.
    pub base: Option<AlignedBoxWithSlice<T>>,
    pub num_points: usize,
}

/// Replayable experiment inputs. Ordinary search uses LoadedBase + QuerySource.
pub struct LoadedDataset<T = f32> {
    pub base_data: LoadedBase<T>,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Option<Vec<Vec<u32>>>,
}

impl<T> std::ops::Deref for LoadedDataset<T> {
    type Target = LoadedBase<T>;
    fn deref(&self) -> &Self::Target { &self.base_data }
}
impl<T> std::ops::DerefMut for LoadedDataset<T> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.base_data }
}

impl<T: BaseElement> LoadedBase<T> {
    /// Transfers ownership of the base-vector allocation into an
    /// [`InmemDataset`](diskann::model::InmemDataset).
    ///
    /// The underlying allocation is moved rather than copied. After a
    /// successful call, [`Self::base`] is `None`, so the allocation cannot be
    /// transferred a second time.
    ///
    /// # Errors
    ///
    /// Returns an error if the base allocation has already been transferred.
    pub fn take_index_dataset<const N: usize>(
        &mut self,
    ) -> Result<diskann::model::InmemDataset<T, N>, String>
    where
        [T; N]: vector::FullPrecisionDistance<T, N>,
    {
        let allocation = self
            .base
            .take()
            .ok_or("base allocation already transferred")?;
        Ok(diskann::model::InmemDataset {
            data: allocation,
            num_points: self.num_points,
            num_active_pts: self.num_points,
            capacity: self.num_points * N,
        })
    }

    /// Loads the benchmark dataset described by `config` into memory.
    ///
    /// The method validates the dataset metadata and small auxiliary inputs
    /// before allocating the potentially large base-vector buffer. Base
    /// vectors are loaded into a SIMD-aligned allocation and normalized
    /// according to the configured distance metric.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured storage type does not match `T`,
    /// dataset dimensions are inconsistent, the point count is invalid,
    /// memory-budget validation fails, allocation fails, or any input file
    /// cannot be loaded or validated.
    pub fn load(config: &ResolvedRunConfig) -> Result<Self, String> {
        if config.search_plan().storage().kind != T::STORAGE {
            return Err("this executable expects f32 storage; use the orion binary for native u8, or --vector-storage f32 --admission l2-u8".into());
        }
        let base_header = inspect_vector_file(&config.base, config.base_format)?;
        if base_header.dimension != config.dimension {
            return Err(format!("base dimension {} != configured dimension {}", base_header.dimension, config.dimension));
        }

        let num_points = base_header.count.min(config.max_points);
        if num_points == 0 || num_points > u32::MAX as usize || (!config.prepare_only && config.sweep.k > num_points) {
            return Err("invalid point count or k (IDs must fit u32 excluding sentinel)".into());
        }
        let dimension = base_header.dimension;

        super::resources::check_memory_budget(config, num_points)?;

        let coordinate_count = num_points
            .checked_mul(dimension)
            .ok_or("base size overflow")?;
        // Reserve one tail for SIMD reads past the final vector. Individual
        // vectors remain contiguous and keep their original dimension.
        let allocation_elements = coordinate_count
            .checked_add(SIMD_ALIGNMENT_BYTES / std::mem::size_of::<T>())
            .ok_or("base size overflow")?;
        let mut base = AlignedBoxWithSlice::new(allocation_elements, SIMD_ALIGNMENT_BYTES)
            .map_err(|e| e.to_string())?;

        read_vectors_into(
            &config.base,
            config.base_format,
            dimension,
            &mut base[..coordinate_count],
        )?;
        T::normalize(&mut base[..coordinate_count], dimension, config.metric)?;
        Ok(Self {
            base: Some(base),
            num_points,
        })
    }
}

impl<T: BaseElement> LoadedDataset<T> {
    pub fn load(config: &ResolvedRunConfig) -> Result<Self, String> {
        if config.prepare_only {
            return Ok(Self { base_data: LoadedBase::load(config)?, queries: vec![], ground_truth: None });
        }
        let path = config.query.as_deref().ok_or("--query is required")?;
        if path == Path::new("-") || !std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.is_file() {
            return Err("sweep/diagnostics require a replayable query file; use orion for streaming queries".into());
        }
        let report = super::resources::preflight_report(config)?;
        super::resources::enforce_budget(&report)?;
        let mut source = super::query::open_queries(config)?;
        let mut batch = Vec::new();
        let mut queries = Vec::new();
        while source.read_batch(&mut batch, 1024)? != 0 {
            queries.extend(batch.chunks_exact(config.dimension).map(|row| row.to_vec()));
        }
        let points = inspect_vector_file(&config.base, config.base_format)?.count.min(config.max_points);
        let ground_truth = if let Some(path) = &config.groundtruth {
            let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let mut reader = super::query::GroundTruthReader::new(BufReader::new(file), config.sweep.k, points);
            let mut rows = Vec::with_capacity(queries.len());
            for _ in &queries {
                rows.push(reader.next()?.ok_or("ground truth has fewer rows than queries")?);
            }
            if reader.next()?.is_some() { return Err("ground truth has more rows than queries".into()); }
            Some(rows)
        } else { None };
        Ok(Self { base_data: LoadedBase::load(config)?, queries, ground_truth })
    }
}

/// Decode up to `limit` records with four-byte elements (for example, ivecs IDs).
/// Validate every consumed dimension header; a valid first row is not sufficient.
#[cfg(test)]
fn read_dimensioned_records<R: Read + Seek, T>(
    reader: &mut R,
    limit: usize,
    decode: fn([u8; 4]) -> T,
) -> Result<(Vec<T>, usize), String> {
    let bytes = reader.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    let mut header = [0; 4];
    reader.read_exact(&mut header).map_err(|e| e.to_string())?;
    let dimension = u32::from_le_bytes(header) as usize;
    if dimension == 0 || limit == 0 {
        return Err("empty vectors or zero record limit".into());
    }
    let record_bytes = (dimension as u64 + 1) * 4;
    if bytes % record_bytes != 0 {
        return Err("truncated or inconsistent vector records".into());
    }
    let count = usize::try_from(bytes / record_bytes)
        .map_err(|e| e.to_string())?
        .min(limit);

    let element_count = count.checked_mul(dimension).ok_or("vector size overflow")?;
    let record_bytes = dimension.checked_mul(4).ok_or("dimension overflow")?;
    let mut data = Vec::with_capacity(element_count);
    let mut record = vec![0; record_bytes];

    reader.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
    for _ in 0..count {
        reader.read_exact(&mut header).map_err(|e| e.to_string())?;
        if u32::from_le_bytes(header) as usize != dimension {
            return Err("inconsistent per-record dimension".into());
        }
        reader.read_exact(&mut record).map_err(|e| e.to_string())?;
        data.extend(
            record
                .chunks_exact(4)
                .map(|c| decode(c.try_into().unwrap())),
        );
    }
    Ok((data, dimension))
}

// Normalize only cosine inputs. Raw inner-product searches depend on vector norms.
pub(crate) fn validate_and_normalize_vectors(
    data: &mut [f32],
    dimension: usize,
    metric: SearchMetric,
) -> Result<(), String> {
    for vector in data.chunks_exact_mut(dimension) {
        if vector.iter().any(|v| !v.is_finite()) {
            return Err("non-finite vector coordinate".into());
        }
        if metric == SearchMetric::Cosine {
            let norm = vector
                .iter()
                .map(|&v| (v as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            if norm == 0.0 {
                return Err("cosine is undefined for zero vectors".into());
            }
            for value in vector {
                *value = (*value as f64 / norm) as f32;
            }
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn bytes(rows: &[(u32, [f32; 2])]) -> Vec<u8> {
        let mut out = Vec::new();
        for (d, values) in rows {
            out.extend(d.to_le_bytes());
            for v in values {
                out.extend(v.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn records_validate_headers_and_respect_limit() {
        let data = bytes(&[(2, [1.0, 2.0]), (2, [3.0, 4.0])]);
        let (read, d) =
            read_dimensioned_records(&mut Cursor::new(data.clone()), 1, f32::from_le_bytes)
                .unwrap();
        assert_eq!((read, d), (vec![1.0, 2.0], 2));
        let invalid = bytes(&[(2, [1.0, 2.0]), (3, [3.0, 4.0])]);
        assert!(read_dimensioned_records(
            &mut Cursor::new(invalid),
            usize::MAX,
            f32::from_le_bytes
        )
            .is_err());
        assert!(read_dimensioned_records(
            &mut Cursor::new(&data[..data.len() - 1]),
            usize::MAX,
            f32::from_le_bytes
        )
            .is_err());
    }

    #[test]
    fn cosine_normalizes_but_raw_mips_keeps_norms() {
        let mut raw = [3.0, 4.0, 6.0, 8.0];
        let original = raw;
        validate_and_normalize_vectors(&mut raw, 2, SearchMetric::InnerProduct).unwrap();
        assert_eq!(raw, original);
        validate_and_normalize_vectors(&mut raw, 2, SearchMetric::Cosine).unwrap();
        assert_eq!(raw, [0.6, 0.8, 0.6, 0.8]);
        assert!(validate_and_normalize_vectors(&mut [0.0, 0.0], 2, SearchMetric::Cosine).is_err());
        assert!(validate_and_normalize_vectors(&mut [f32::NAN, 0.0], 2, SearchMetric::L2).is_err());
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn path() -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "orion-vectors-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    #[test]
    fn formats_decode_identically_and_respect_prefix() {
        for format in [
            VectorFormat::Fvecs,
            VectorFormat::Bvecs,
            VectorFormat::Fbin,
            VectorFormat::U8bin,
        ] {
            let p = path();
            let mut b = Vec::new();
            if format.has_file_header() {
                b.extend(2u32.to_le_bytes());
                b.extend(2u32.to_le_bytes());
            }
            for row in [[1u8, 2], [3, 4]] {
                if !format.has_file_header() {
                    b.extend(2u32.to_le_bytes());
                }
                for x in row {
                    if format.coordinate_bytes() == 1 {
                        b.push(x);
                    } else {
                        b.extend((x as f32).to_le_bytes());
                    }
                }
            }
            std::fs::write(&p, &b).unwrap();
            let h = inspect_vector_file(&p, format).unwrap();
            assert_eq!((h.count, h.dimension), (2, 2));
            let mut all = [0.; 4];
            read_vectors_into(&p, format, 2, &mut all).unwrap();
            assert_eq!(all, [1., 2., 3., 4.]);
            let mut prefix = [0.; 2];
            read_vectors_into(&p, format, 2, &mut prefix).unwrap();
            assert_eq!(prefix, [1., 2.]);
            std::fs::write(&p, &b[..b.len() - 1]).unwrap();
            assert!(inspect_vector_file(&p, format).is_err());
            std::fs::remove_file(p).unwrap();
        }
    }
    #[test]
    fn sparse_billion_header_does_not_allocate_or_truncate_to_u32_bytes() {
        use std::io::Write;
        let p = path();
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(&1_000_000_000u32.to_le_bytes()).unwrap();
        f.write_all(&128u32.to_le_bytes()).unwrap();
        f.set_len(128_000_000_008).unwrap();
        drop(f);
        let h = inspect_vector_file(&p, VectorFormat::U8bin).unwrap();
        assert_eq!((h.count, h.dimension), (1_000_000_000, 128));
        std::fs::remove_file(p).unwrap();
    }
}
