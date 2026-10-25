//! Bounded-memory input preparation for the f32 DiskANN baseline.
use orion_cli::cascade::SearchMetric;
use orion_cli::cli::data::{inspect_vector_file, VectorFormat};
use orion_cli::cli::query::{GroundTruthReader, QuerySource, VectorReader};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

const BATCH_ROWS: usize = 8192;

fn reader(
    path: &Path,
    format: VectorFormat,
    dim: usize,
) -> anyhow::Result<VectorReader<BufReader<File>>> {
    VectorReader::new(
        BufReader::new(File::open(path)?),
        format,
        dim,
        SearchMetric::L2,
    )
    .map_err(anyhow::Error::msg)
}

/// Owns only a temporary conversion file. An existing fbin is borrowed by path.
/// Unwinding on a build error also removes the temporary file; SIGKILL cannot.
pub(super) struct BuildInput {
    path: PathBuf,
    temporary: bool,
}

impl BuildInput {
    pub(super) fn path(&self) -> &Path {
        &self.path
    }

    pub(super) fn prepare(
        source: &Path,
        format: VectorFormat,
        count: usize,
        dim: usize,
        normalize: bool,
        scratch_dir: &Path,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            count > 0 && count <= i32::MAX as usize,
            "invalid base count"
        );
        anyhow::ensure!(dim > 0 && dim <= i32::MAX as usize, "invalid dimension");
        let shape = inspect_vector_file(source, format).map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            shape.dimension == dim && shape.count >= count,
            "base shape mismatch"
        );
        // DiskANN rejects a file whose declared count exceeds index capacity,
        // so a prefix of an fbin must also be rewritten with the smaller header.
        if format == VectorFormat::Fbin && !normalize && count == shape.count {
            return Ok(Self {
                path: source.to_owned(),
                temporary: false,
            });
        }

        let path = scratch_dir.join(format!("diskann-build-{}.fbin", std::process::id()));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let temporary = Self {
            path,
            temporary: true,
        };
        let mut output = BufWriter::with_capacity(1024 * 1024, file);
        output.write_all(&(count as u32).to_le_bytes())?;
        output.write_all(&(dim as u32).to_le_bytes())?;
        let mut source = reader(source, format, dim)?;
        let mut batch = Vec::new();
        let mut encoded = Vec::new();
        let mut remaining = count;
        while remaining > 0 {
            let rows = source
                .read_batch(&mut batch, remaining.min(BATCH_ROWS))
                .map_err(anyhow::Error::msg)?;
            anyhow::ensure!(rows > 0, "base truncated during conversion");
            if normalize {
                super::l2_normalize_in_place(&mut batch, dim);
            }
            encoded.clear();
            encoded.extend(batch.iter().flat_map(|value| value.to_le_bytes()));
            output.write_all(&encoded)?;
            remaining -= rows;
        }
        output.flush()?;
        Ok(temporary)
    }
}

impl Drop for BuildInput {
    fn drop(&mut self) {
        if self.temporary {
            if let Err(error) = std::fs::remove_file(&self.path) {
                eprintln!("Cannot remove temporary {}: {error}", self.path.display());
            }
        }
    }
}

pub(super) fn load_queries(
    path: &Path,
    truth: &Path,
    dim: usize,
    count: usize,
    k: usize,
    normalize: bool,
) -> anyhow::Result<(Vec<Vec<f32>>, Vec<Vec<u32>>)> {
    let format = VectorFormat::infer(path).map_err(anyhow::Error::msg)?;
    let mut source = reader(path, format, dim)?;
    let mut queries = Vec::new();
    let mut batch = Vec::new();
    while source
        .read_batch(&mut batch, BATCH_ROWS)
        .map_err(anyhow::Error::msg)?
        > 0
    {
        if normalize {
            super::l2_normalize_in_place(&mut batch, dim);
        }
        queries.extend(batch.chunks_exact(dim).map(<[f32]>::to_vec));
    }
    anyhow::ensure!(
        !queries.is_empty() && k <= count,
        "empty queries or k exceeds base count"
    );
    let mut source = GroundTruthReader::new(BufReader::new(File::open(truth)?), k, count);
    let mut gt = Vec::new();
    while let Some(row) = source.next().map_err(anyhow::Error::msg)? {
        gt.push(row);
    }
    anyhow::ensure!(
        gt.len() == queries.len(),
        "query and ground truth counts differ"
    );
    Ok((queries, gt))
}

/// Metadata identity avoids another full read of a 1B input on cache hits.
/// This is not a content checksum: replacing bytes while preserving timestamps
/// requires removing the cache. Use a separate cache directory per experiment.
#[allow(clippy::too_many_arguments)]
pub(super) fn cache_identity(
    path: &Path,
    count: usize,
    dim: usize,
    normalize: bool,
    degree: u32,
    build_l: usize,
    alpha: f32,
    metric: &str,
) -> anyhow::Result<Vec<u8>> {
    let metadata = path.metadata()?;
    Ok(serde_json::to_vec_pretty(&serde_json::json!({
        "version": 1, "storage": "f32", "source": path.canonicalize()?,
        "source_bytes": metadata.len(),
        "source_modified": metadata.modified()?.duration_since(std::time::UNIX_EPOCH)?.as_nanos().to_string(),
        "count": count, "dimension": dim, "normalize": normalize,
        "degree": degree, "build_l": build_l, "alpha": alpha, "metric": metric,
    }))?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQUENCE: AtomicUsize = AtomicUsize::new(0);

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "diskann-input-test-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            path
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn decode(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap();
        bytes[8..]
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn converts_byte_prefix_and_removes_only_temporary() {
        let fixture = Fixture::new();
        let source = fixture.write("base.bvecs", &[2, 0, 0, 0, 3, 4, 2, 0, 0, 0, 255, 0]);
        let input =
            BuildInput::prepare(&source, VectorFormat::Bvecs, 1, 2, false, &fixture.0).unwrap();
        assert_eq!(decode(input.path()), [3.0, 4.0]);
        let temporary = input.path().to_owned();
        drop(input);
        assert!(!temporary.exists());
        assert!(source.exists());
    }

    #[test]
    fn direct_fbin_is_not_deleted_and_normalization_streams() {
        let fixture = Fixture::new();
        let mut bytes = vec![1, 0, 0, 0, 2, 0, 0, 0];
        bytes.extend([3f32, 4f32].iter().flat_map(|v| v.to_le_bytes()));
        let source = fixture.write("base.fbin", &bytes);
        let input =
            BuildInput::prepare(&source, VectorFormat::Fbin, 1, 2, false, &fixture.0).unwrap();
        assert_eq!(input.path(), source);
        drop(input);
        assert!(source.exists());
        let normalized =
            BuildInput::prepare(&source, VectorFormat::Fbin, 1, 2, true, &fixture.0).unwrap();
        assert_eq!(decode(normalized.path()), [0.6, 0.8]);
    }

    #[test]
    fn conversion_crosses_batch_boundary_and_rewrites_fbin_prefix() {
        let fixture = Fixture::new();
        let rows = BATCH_ROWS + 3;
        let mut bytes = Vec::new();
        for row in 0..rows {
            bytes.extend(1u32.to_le_bytes());
            bytes.extend((row as f32).to_le_bytes());
        }
        let source = fixture.write("base.fvecs", &bytes);
        let input =
            BuildInput::prepare(&source, VectorFormat::Fvecs, rows, 1, false, &fixture.0).unwrap();
        let values = decode(input.path());
        assert_eq!(values.len(), rows);
        assert_eq!(values[rows - 1], (rows - 1) as f32);
        let fbin = fixture.write("base.fbin", &std::fs::read(input.path()).unwrap());
        drop(input);
        let prefix =
            BuildInput::prepare(&fbin, VectorFormat::Fbin, 2, 1, false, &fixture.0).unwrap();
        assert_ne!(prefix.path(), fbin);
        assert_eq!(decode(prefix.path()), [0.0, 1.0]);
        assert_eq!(
            &std::fs::read(prefix.path()).unwrap()[..4],
            &2u32.to_le_bytes()
        );
    }

    #[test]
    fn malformed_later_row_cleans_partial_output() {
        let fixture = Fixture::new();
        let source = fixture.write("bad.bvecs", &[2, 0, 0, 0, 3, 4, 3, 0, 0, 0, 1, 2]);
        assert!(
            BuildInput::prepare(&source, VectorFormat::Bvecs, 2, 2, false, &fixture.0).is_err()
        );
        assert_eq!(std::fs::read_dir(&fixture.0).unwrap().count(), 1);
    }

    #[test]
    fn reader_does_not_consume_unrequested_rows() {
        let bytes = [2, 0, 0, 0, 3, 4, 255, 255, 255, 255, 0, 0];
        let mut reader =
            VectorReader::new(Cursor::new(bytes), VectorFormat::Bvecs, 2, SearchMetric::L2)
                .unwrap();
        let mut batch = Vec::new();
        assert_eq!(reader.read_batch(&mut batch, 1).unwrap(), 1);
        assert_eq!(batch, [3.0, 4.0]);
    }
}
