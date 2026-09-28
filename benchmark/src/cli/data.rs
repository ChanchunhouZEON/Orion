//! Bounded-buffer vector loading into the allocation transferred to Orion.
use super::config::ResolvedRunConfig;
use crate::cascade::SearchMetric;
use diskann::common::AlignedBoxWithSlice;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

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
    fn width(self) -> usize {
        if matches!(self, Self::Bvecs | Self::U8bin) {
            1
        } else {
            4
        }
    }
    fn bin(self) -> bool {
        matches!(self, Self::Fbin | Self::U8bin)
    }
}

#[derive(Debug, serde::Serialize)]
pub struct VectorHeader {
    pub count: usize,
    pub dimension: usize,
}
pub fn inspect(path: &Path, format: VectorFormat) -> Result<VectorHeader, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let bytes = f.metadata().map_err(|e| e.to_string())?.len();
    let word = |f: &mut std::fs::File| -> Result<usize, String> {
        let mut b = [0; 4];
        f.read_exact(&mut b).map_err(|e| e.to_string())?;
        Ok(u32::from_le_bytes(b) as usize)
    };
    let first = word(&mut f)?;
    let (count, dimension) = if format.bin() {
        (first, word(&mut f)?)
    } else {
        let record = 4 + first as u64 * format.width() as u64;
        if bytes % record != 0 {
            return Err("truncated vector records".into());
        }
        (
            usize::try_from(bytes / record).map_err(|e| e.to_string())?,
            first,
        )
    };
    if dimension == 0 || count == 0 {
        return Err("empty vectors".into());
    }
    let expected = if format.bin() {
        8u64.checked_add(
            (count as u64)
                .checked_mul(dimension as u64)
                .and_then(|x| x.checked_mul(format.width() as u64))
                .ok_or("file size overflow")?,
        )
        .ok_or("file size overflow")?
    } else {
        bytes
    };
    if expected != bytes {
        return Err("binary header/file length mismatch".into());
    }
    Ok(VectorHeader { count, dimension })
}

fn read_vectors_into(
    path: &Path,
    format: VectorFormat,
    dimension: usize,
    out: &mut [f32],
) -> Result<(), String> {
    let mut r = BufReader::with_capacity(
        8 * 1024 * 1024,
        std::fs::File::open(path).map_err(|e| e.to_string())?,
    );
    if format.bin() {
        r.seek(SeekFrom::Start(8)).map_err(|e| e.to_string())?;
    }
    let mut row = vec![
        0;
        dimension
            .checked_mul(format.width())
            .ok_or("row overflow")?
    ];
    for values in out.chunks_exact_mut(dimension) {
        if !format.bin() {
            let mut h = [0; 4];
            r.read_exact(&mut h).map_err(|e| e.to_string())?;
            if u32::from_le_bytes(h) as usize != dimension {
                return Err("inconsistent per-record dimension".into());
            }
        }
        r.read_exact(&mut row).map_err(|e| e.to_string())?;
        if format.width() == 1 {
            for (v, &b) in values.iter_mut().zip(&row) {
                *v = b as f32;
            }
        } else {
            for (v, b) in values.iter_mut().zip(row.chunks_exact(4)) {
                *v = f32::from_le_bytes(b.try_into().unwrap());
            }
        }
    }
    Ok(())
}

pub struct LoadedDataset {
    // Includes the same 64-byte trailing SIMD pad as InmemDataset.
    pub base: Option<AlignedBoxWithSlice<f32>>,
    pub num_points: usize,
    pub queries: Vec<Vec<f32>>,
    pub ground_truth: Vec<Vec<u32>>,
}
impl LoadedDataset {
    pub fn load(config: &ResolvedRunConfig) -> Result<Self, String> {
        let h = inspect(&config.base, config.base_format)?;
        let qh = inspect(&config.query, config.query_format)?;
        if h.dimension != config.dimension || qh.dimension != h.dimension {
            return Err("base/query/config dimension mismatch".into());
        }
        let num_points = h.count.min(config.max_points);
        if num_points == 0 || num_points > u32::MAX as usize || config.sweep.k > num_points {
            return Err("invalid point count or k (IDs must fit u32 excluding sentinel)".into());
        }
        let dim = h.dimension;
        // Validate small inputs and GT before allocating the large base.
        let mut queries = vec![0.; qh.count.checked_mul(dim).ok_or("query size overflow")?];
        read_vectors_into(&config.query, config.query_format, dim, &mut queries)?;
        prepare_vectors(&mut queries, dim, config.metric)?;
        let queries: Vec<Vec<f32>> = queries.chunks_exact(dim).map(|q| q.to_vec()).collect();
        let (gt, depth) = read_file(&config.groundtruth, queries.len(), u32::from_le_bytes)?;
        let ground_truth: Vec<Vec<u32>> = gt.chunks_exact(depth).map(|r| r.to_vec()).collect();
        crate::utils::validate_ground_truth(&ground_truth, queries.len(), config.sweep.k)?;
        if ground_truth
            .iter()
            .flatten()
            .any(|&id| id as usize >= num_points)
        {
            return Err("ground truth contains IDs outside the loaded base; use ground truth computed for this exact subset".into());
        }
        super::resources::check(config, num_points)?;
        let len = num_points.checked_mul(dim).ok_or("base size overflow")?;
        let mut base =
            AlignedBoxWithSlice::new(len.checked_add(16).ok_or("base size overflow")?, 64)
                .map_err(|e| e.to_string())?;
        read_vectors_into(&config.base, config.base_format, dim, &mut base[..len])?;
        prepare_vectors(&mut base[..len], dim, config.metric)?;
        Ok(Self {
            base: Some(base),
            num_points,
            queries,
            ground_truth,
        })
    }
}

fn read_file<T>(
    path: &Path,
    limit: usize,
    decode: fn([u8; 4]) -> T,
) -> Result<(Vec<T>, usize), String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    read_records(&mut BufReader::new(file), limit, decode)
        .map_err(|e| format!("{}: {e}", path.display()))
}
fn read_records<R: Read + Seek, T>(
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
    let mut data = Vec::with_capacity(count.checked_mul(dimension).ok_or("vector size overflow")?);
    let mut record = vec![0; dimension.checked_mul(4).ok_or("dimension overflow")?];
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
fn prepare_vectors(data: &mut [f32], dimension: usize, metric: SearchMetric) -> Result<(), String> {
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
            read_records(&mut Cursor::new(data.clone()), 1, f32::from_le_bytes).unwrap();
        assert_eq!((read, d), (vec![1.0, 2.0], 2));
        let invalid = bytes(&[(2, [1.0, 2.0]), (3, [3.0, 4.0])]);
        assert!(read_records(&mut Cursor::new(invalid), usize::MAX, f32::from_le_bytes).is_err());
        assert!(read_records(
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
        prepare_vectors(&mut raw, 2, SearchMetric::InnerProduct).unwrap();
        assert_eq!(raw, original);
        prepare_vectors(&mut raw, 2, SearchMetric::Cosine).unwrap();
        assert_eq!(raw, [0.6, 0.8, 0.6, 0.8]);
        assert!(prepare_vectors(&mut [0.0, 0.0], 2, SearchMetric::Cosine).is_err());
        assert!(prepare_vectors(&mut [f32::NAN, 0.0], 2, SearchMetric::L2).is_err());
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
            if format.bin() {
                b.extend(2u32.to_le_bytes());
                b.extend(2u32.to_le_bytes());
            }
            for row in [[1u8, 2], [3, 4]] {
                if !format.bin() {
                    b.extend(2u32.to_le_bytes());
                }
                for x in row {
                    if format.width() == 1 {
                        b.push(x);
                    } else {
                        b.extend((x as f32).to_le_bytes());
                    }
                }
            }
            std::fs::write(&p, &b).unwrap();
            let h = inspect(&p, format).unwrap();
            assert_eq!((h.count, h.dimension), (2, 2));
            let mut all = [0.; 4];
            read_vectors_into(&p, format, 2, &mut all).unwrap();
            assert_eq!(all, [1., 2., 3., 4.]);
            let mut prefix = [0.; 2];
            read_vectors_into(&p, format, 2, &mut prefix).unwrap();
            assert_eq!(prefix, [1., 2.]);
            std::fs::write(&p, &b[..b.len() - 1]).unwrap();
            assert!(inspect(&p, format).is_err());
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
        let h = inspect(&p, VectorFormat::U8bin).unwrap();
        assert_eq!((h.count, h.dimension), (1_000_000_000, 128));
        std::fs::remove_file(p).unwrap();
    }
}
