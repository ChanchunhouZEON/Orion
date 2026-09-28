//! Bounded-buffer vector loading into the allocation transferred to Orion.
use super::config::{ResolvedRunConfig, VectorStorageKind};
use crate::cascade::SearchMetric;
use diskann::common::AlignedBoxWithSlice;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const VECTOR_READ_BUFFER_BYTES: usize = 8 * 1024 * 1024;
const SIMD_ALIGNMENT_BYTES: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, clap::ValueEnum)]
#[serde(rename_all = "kebab-case")]
pub enum VectorFormat {
    Fvecs,
    Bvecs,
    Fbin,
    U8bin,
}

impl VectorFormat {
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

    fn coordinate_bytes(self) -> usize {
        if matches!(self, Self::Bvecs | Self::U8bin) {
            1
        } else {
            4
        }
    }

    fn has_file_header(self) -> bool {
        matches!(self, Self::Fbin | Self::U8bin)
    }
}

#[derive(Debug, serde::Serialize)]
pub struct VectorHeader {
    pub count: usize,
    pub dimension: usize,
}

/// Inspect shape and file size without reading the coordinate payload.
/// Per-record dimensions are checked later while decoding the requested prefix.
pub fn inspect_vector_file(path: &Path, format: VectorFormat) -> Result<VectorHeader, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let file_bytes = file.metadata().map_err(|e| e.to_string())?.len();
    let read_header_word = |file: &mut std::fs::File| -> Result<usize, String> {
        let mut word_bytes = [0; 4];
        file.read_exact(&mut word_bytes)
            .map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(word_bytes) as usize)
    };

    // Binary formats have one [count, dimension] header; vecs formats repeat
    // a dimension word before every vector and derive their count from file size.
    let first_header_word = read_header_word(&mut file)?;
    let (count, dimension) = if format.has_file_header() {
        (first_header_word, read_header_word(&mut file)?)
    } else {
        let record_bytes = 4 + first_header_word as u64 * format.coordinate_bytes() as u64;
        if file_bytes % record_bytes != 0 {
            return Err("truncated vector records".into());
        }
        (
            usize::try_from(file_bytes / record_bytes).map_err(|e| e.to_string())?,
            first_header_word,
        )
    };

    if dimension == 0 || count == 0 {
        return Err("empty vectors".into());
    }

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

/// The loader writes directly into the final storage type. Byte mode rejects
/// float encodings rather than allocating a temporary f32 dataset to quantize.
pub trait BaseElement: Default + Copy + Send + Sync + Into<f32> {
    const STORAGE: VectorStorageKind;
    fn from_byte(value: u8) -> Self;
    fn from_float(value: f32) -> Result<Self, String>;
    fn normalize(data: &mut [Self], dimension: usize, metric: SearchMetric) -> Result<(), String>;
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

fn read_vectors_into<T: BaseElement>(
    path: &Path,
    format: VectorFormat,
    dimension: usize,
    output: &mut [T],
) -> Result<(), String> {
    let mut reader = BufReader::with_capacity(
        VECTOR_READ_BUFFER_BYTES,
        std::fs::File::open(path).map_err(|e| e.to_string())?,
    );
    if format.has_file_header() {
        reader.seek(SeekFrom::Start(8)).map_err(|e| e.to_string())?;
    }

    // Reuse one encoded row: temporary decoding storage is independent of N.
    // The output slice controls how much of the input prefix is read.
    let row_bytes = dimension
        .checked_mul(format.coordinate_bytes())
        .ok_or("row overflow")?;
    let mut encoded_row = vec![0; row_bytes];

    for values in output.chunks_exact_mut(dimension) {
        if !format.has_file_header() {
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
            for (coordinate, &encoded_coordinate) in values.iter_mut().zip(&encoded_row) {
                *coordinate = T::from_byte(encoded_coordinate);
            }
        } else {
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
/// together with the query vectors and ground-truth neighbor IDs required by
/// the benchmark. The base-vector element type is parameterized by `T`, which
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
/// After the transfer, [`base`](Self::base) becomes `None`, while the query
/// vectors and ground truth remain available to the benchmark.
///
/// # Workflow
///
/// A `LoadedDataset` is normally constructed from a validated
/// [`ResolvedRunConfig`] using [`load`](Self::load). The resulting base
/// allocation can then be moved into an
/// [`Orion`](orion::index::compressed_index::Orion) index via
/// [`take_index_dataset`](Self::take_index_dataset).
///
/// This ownership-transfer path avoids keeping a second copy of the original
/// base dataset and eliminates the corresponding `memcpy`, which is especially
/// important for large datasets.
pub struct LoadedDataset<T = f32> {
    /// Padded base allocation. `None` means it has been transferred to the index.
    /// Keep this separate from queries/GT, which the benchmark still needs afterwards.
    pub base: Option<AlignedBoxWithSlice<T>>,
    pub num_points: usize,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<u32>>,
}

impl<T: BaseElement> LoadedDataset<T> {
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
        if config.vector_storage != T::STORAGE {
            return Err("this executable expects f32 storage; use the orion binary for native u8, or --vector-storage f32".into());
        }
        let base_header = inspect_vector_file(&config.base, config.base_format)?;
        let query_header = inspect_vector_file(&config.query, config.query_format)?;
        if base_header.dimension != config.dimension
            || query_header.dimension != base_header.dimension
        {
            return Err("base/query/config dimension mismatch".into());
        }

        let num_points = base_header.count.min(config.max_points);
        if num_points == 0 || num_points > u32::MAX as usize || config.sweep.k > num_points {
            return Err("invalid point count or k (IDs must fit u32 excluding sentinel)".into());
        }
        let dimension = base_header.dimension;

        // Validate small inputs and GT before allocating the large base.
        let queries = load_queries(config, query_header.count, dimension)?;
        let ground_truth = load_ground_truth(config, queries.len(), num_points)?;
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
            queries,
            ground_truth,
        })
    }
}

fn load_queries(
    config: &ResolvedRunConfig,
    query_count: usize,
    dimension: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let coordinate_count = query_count
        .checked_mul(dimension)
        .ok_or("query size overflow")?;
    let mut queries = vec![0.; coordinate_count];

    read_vectors_into(&config.query, config.query_format, dimension, &mut queries)?;
    validate_and_normalize_vectors(&mut queries, dimension, config.metric)?;

    let queries: Vec<Vec<f32>> = queries
        .chunks_exact(dimension)
        .map(|q| q.to_vec())
        .collect();
    Ok(queries)
}

fn load_ground_truth(
    config: &ResolvedRunConfig,
    query_count: usize,
    num_points: usize,
) -> Result<Vec<Vec<u32>>, String> {
    let (ground_truth_values, neighbors_per_query) =
        read_dimensioned_file(&config.groundtruth, query_count, u32::from_le_bytes)?;
    let ground_truth: Vec<Vec<u32>> = ground_truth_values
        .chunks_exact(neighbors_per_query)
        .map(|r| r.to_vec())
        .collect();
    crate::utils::validate_ground_truth(&ground_truth, query_count, config.sweep.k)?;
    if ground_truth
        .iter()
        .flatten()
        .any(|&id| id as usize >= num_points)
    {
        return Err("ground truth contains IDs outside the loaded base; use ground truth computed for this exact subset".into());
    }
    Ok(ground_truth)
}

fn read_dimensioned_file<T>(
    path: &Path,
    limit: usize,
    decode: fn([u8; 4]) -> T,
) -> Result<(Vec<T>, usize), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    read_dimensioned_records(&mut BufReader::new(file), limit, decode)
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Decode up to `limit` records with four-byte elements (for example, ivecs IDs).
/// Validate every consumed dimension header; a valid first row is not sufficient.
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
fn validate_and_normalize_vectors(
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
