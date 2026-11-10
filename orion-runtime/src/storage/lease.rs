/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
//! Short, nonblocking cross-process writer leases. Readers do not acquire them.
use crate::{Error, Result};
use std::{
    fs::{File, OpenOptions},
    path::Path,
};

pub(crate) struct WriterLease {
    _file: File,
}
impl WriterLease {
    pub(crate) fn acquire(destination: &Path) -> Result<Self> {
        let absolute = std::path::absolute(destination)?;
        let parent = absolute
            .parent()
            .ok_or_else(|| Error::invalid_argument("writer destination needs a parent"))?;
        std::fs::create_dir_all(parent)?;
        let absolute = match absolute.canonicalize() {
            Ok(path) => path,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                parent.canonicalize()?.join(
                    absolute.file_name().ok_or_else(|| {
                        Error::invalid_argument("writer destination needs a name")
                    })?,
                )
            }
            Err(error) => return Err(Error::io("resolve writer destination", &absolute, error)),
        };
        let mut name = std::ffi::OsString::from(".");
        name.push(
            absolute
                .file_name()
                .ok_or_else(|| Error::invalid_argument("writer destination needs a name"))?,
        );
        name.push(".writer.lock");
        let lock = absolute.with_file_name(name);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock)?;
        file.try_lock().map_err(|error| match error {
            std::fs::TryLockError::WouldBlock => {
                Error::invalid_state("another writer owns this destination")
                    .with_detail("path", absolute.display().to_string())
            }
            std::fs::TryLockError::Error(error) => Error::io("acquire writer lease", &lock, error),
        })?;
        // Keep the lock inode: unlinking it allows a third writer to bypass an
        // existing lease. File closure releases the OS lock, including on exit.
        Ok(Self { _file: file })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conflicting_writers_fail_and_drop_releases_lease() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("cache");
        let first = WriterLease::acquire(&target).unwrap();
        assert!(WriterLease::acquire(&target).is_err());
        drop(first);
        assert!(WriterLease::acquire(&target).is_ok());
    }
}
