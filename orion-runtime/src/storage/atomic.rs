/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Same-filesystem publication and cleanup on failure.
use crate::{Error, Result};
use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
/// Each writer owns its temporary file, including cleanup after a failed encode.
pub(crate) struct PendingFile(
    /// Unpublished temporary path removed when this guard is dropped.
    pub(crate) PathBuf,
);

impl PendingFile {
    /// Creates a unique sibling file without truncating another writer's temporary output.
    pub(crate) fn create(destination: &Path) -> Result<(Self, File)> {
        static SEQUENCE: AtomicU64 = AtomicU64::new(0);
        loop {
            let id = SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let name = format!(".orion-{}-{id}.tmp", std::process::id());
            let path = destination.with_file_name(name);
            match File::create_new(&path) {
                Ok(file) => return Ok((Self(path), file)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(Error::io("create", &path, error)),
            }
        }
    }
}

impl Drop for PendingFile {
    /// Best-effort removes this writer's temporary name after failure or publication.
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

/// Requests directory-entry durability on Unix; other targets perform no operation.
pub(crate) fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    fs::File::open(path)?.sync_all()?;
    #[cfg(not(unix))]
    let _ = path;
    Ok(())
}

/// Builds a generation in a sibling directory and publishes it with one rename.
///
/// A writer lease prevents competing publication. Failures clean up the staging
/// directory and do not replace an existing destination.
pub(crate) fn publish<T>(destination: &Path, write: impl FnOnce(&Path) -> Result<T>) -> Result<T> {
    let _lease = super::WriterLease::acquire(destination)?;
    if destination.try_exists()? {
        return Err(Error::invalid_state(
            "publication destination already exists",
        ));
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(Error::from)?;
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let staging = parent.join(format!(
        ".orion-save-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut directory = fs::DirBuilder::new();
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut directory, 0o700);
    directory.create(&staging)?;
    let result = (|| {
        let value = write(&staging)?;
        sync_dir(&staging)?;
        fs::rename(&staging, destination)?;
        sync_dir(parent)?;
        Ok(value)
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(staging);
    }
    result
}

/// Verifies a failed staged write leaves the previous generation intact and no new directory visible.
#[test]
fn incomplete_generation_is_never_published() {
    let root = std::env::temp_dir().join(format!("orion-publish-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let previous = root.join("old");
    publish(&previous, |p| Ok(fs::write(p.join("index.json"), b"old")?)).unwrap();
    let next = root.join("next");
    let failed: Result<()> = publish(&next, |p| {
        fs::write(p.join("graph"), b"partial")?;
        assert_eq!(fs::read(previous.join("index.json"))?, b"old");
        Err(Error::invalid_state("injected write failure"))
    });
    assert!(failed.is_err());
    assert!(!next.exists());
    assert_eq!(
        fs::read_dir(&root)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().unwrap().is_dir())
            .count(),
        1
    );
    fs::remove_dir_all(root).unwrap();
}
