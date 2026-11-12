/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Benchmark-owned sidecars: decode legacy files or CAS objects without hashing.
//! This service deliberately produces no verification receipts for snapshots.
use super::working::{metadata_path, Identity};
use crate::{Error, Result};
use orion_ann_core::{
    index::{DecodeSidecar, EncodeSidecar, IndexData, SidecarCache},
    model::dataset::rows::Rows,
    CompressedIndex,
};
use orion_ann_platform::buffered_io::FILE_WRITE_BYTES;
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

struct TrustedCache {
    metadata: PathBuf,
    objects: PathBuf,
}

pub(crate) fn attach<D: IndexData>(index: &mut CompressedIndex<D>, path: &Path) -> Result<()> {
    if index.dataset().element_type() == "f32" {
        index.set_sidecar_cache(Box::new(TrustedCache {
            metadata: path.to_owned(),
            objects: crate::storage::object_dir(path.parent().unwrap_or(Path::new(".")))?,
        }));
    }
    Ok(())
}

impl SidecarCache for TrustedCache {
    fn load(
        &self,
        _base: &dyn Rows,
        codec: &'static str,
        decode: &mut DecodeSidecar<'_>,
    ) -> Result<()> {
        let path = self.metadata.with_extension(codec);
        // Flat payloads take precedence only in this explicitly trusted service.
        let source = if path.try_exists()? {
            path
        } else {
            let record = metadata_path(&path);
            let file = match File::open(&record) {
                Ok(file) => file,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(e) => return Err(Error::io("open benchmark sidecar record", &record, e)),
            };
            let bytes = crate::storage::read_bounded_file(file, &record, 4096)?;
            let identity: Identity = serde_json::from_slice(&bytes)?;
            crate::storage::object(&self.objects, &identity.payload_sha256, codec)?
        };
        let file = File::open(&source)?;
        let bytes = file.metadata()?.len();
        let mut reader = BufReader::with_capacity(FILE_WRITE_BYTES, file);
        decode(&mut reader, bytes)?;
        if reader.read(&mut [0])? != 0 {
            return Err(Error::cache("trailing benchmark sidecar bytes"));
        }
        Ok(())
    }

    fn store(
        &self,
        _base: &dyn Rows,
        codec: &'static str,
        encode: &EncodeSidecar<'_>,
    ) -> Result<()> {
        // New benchmark-only payloads stay flat. Never publish an identity record
        // that could be mistaken for authenticated production cache contents.
        let path = self.metadata.with_extension(codec);
        let (temporary, file) = crate::storage::PendingFile::create(&path)?;
        let mut writer = BufWriter::with_capacity(FILE_WRITE_BYTES, file);
        encode(&mut writer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&temporary.0, &path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_and_cas_sidecars_skip_hashes_but_keep_decoder_checks() {
        let root = std::env::temp_dir().join(format!("trusted-sidecar-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let cache = TrustedCache {
            metadata: root.join("index.bin"),
            objects: root.join("objects"),
        };
        fs::create_dir(&cache.objects).unwrap();
        let base = orion_ann_core::PaddedDataset::from_packed(
            &[0.0f32, 1.],
            orion_ann_core::PaddedLayout::for_kernel::<orion_ann_vector::L2F32Distance>(2).unwrap(),
        )
        .unwrap();
        let payload = cache.metadata.with_extension("qds");
        cache
            .store(&base, "qds", &|w| {
                w.write_all(b"fixture")?;
                Ok(())
            })
            .unwrap();
        let mut read = |reader: &mut dyn Read, bytes: u64| -> Result<()> {
            assert_eq!(bytes, 7);
            let mut content = Vec::new();
            reader.read_to_end(&mut content)?;
            assert_eq!(content, b"fixture");
            Ok(())
        };
        cache.load(&base, "qds", &mut read).unwrap();
        assert!(!metadata_path(&payload).exists());
        // CAS lookup uses the digest as an address only, with no input or payload scan.
        let digest = "0".repeat(64);
        fs::rename(&payload, cache.objects.join(format!("{digest}.qds"))).unwrap();
        fs::write(
            metadata_path(&payload),
            serde_json::to_vec(&Identity {
                dataset_sha256: "deliberately-unmatched".into(),
                payload_sha256: digest,
            })
            .unwrap(),
        )
        .unwrap();
        cache.load(&base, "qds", &mut read).unwrap();
        assert!(cache
            .load(&base, "qds", &mut |_, _| Err(Error::cache("invalid shape")))
            .is_err());
        assert!(
            cache.load(&base, "qds", &mut |_, _| Ok(())).is_err(),
            "trailing bytes must fail"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
