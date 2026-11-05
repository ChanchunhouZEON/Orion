/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Compute and verify content identities on the stream passed to the decoder.
use crate::{Error, Result};
use orion_ann_platform::buffered_io::FILE_WRITE_BYTES;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufReader, Read, Write},
    path::Path,
};
/// Evidence about the bytes consumed by a successful decoder on a single handle.
pub(crate) struct Verified<T> {
    pub value: T,
    pub bytes: u64,
    pub digest: String,
}

pub(crate) struct HashedFile {
    file: File,
    hash: Sha256,
}

pub(crate) fn read_verified<T>(
    path: &Path,
    role: &str,
    expected: &str,
    expected_bytes: Option<u64>,
    decode: impl FnOnce(&mut dyn Read, u64) -> Result<T>,
) -> Result<Verified<T>> {
    let file = File::open(path).map_err(|e| Error::io("open", path, e))?;
    let bytes = file.metadata()?.len();
    if expected_bytes.is_some_and(|expected| expected != bytes) {
        return Err(Error::cache(format!("{role} length mismatch")).with_detail("role", role));
    }
    let mut source = HashedFile::new(file);
    let mut reader = BufReader::with_capacity(FILE_WRITE_BYTES, &mut source);
    let value =
        decode(&mut reader, bytes).map_err(|e| e.with_detail("path", path.to_string_lossy()))?;
    let trailing = reader.read(&mut [0])? != 0;
    let (_, digest) = source.finish();
    if trailing || digest != expected {
        return Err(
            Error::cache(format!("snapshot {role} SHA-256 mismatch")).with_detail("role", role)
        );
    }
    Ok(Verified {
        value,
        bytes,
        digest,
    })
}

pub(crate) fn file_sha256(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|e| Error::io("hash", path, e))?;
    Ok(hash_file(file)?.1)
}

/// Return the number of bytes actually hashed, even if the path is replaced.
pub(crate) fn hash_file(file: File) -> Result<(u64, String)> {
    let block_bytes = file.metadata()?.len().clamp(1, FILE_WRITE_BYTES as u64) as usize;
    let mut source = HashedFile::new(file);
    let mut block = vec![0; block_bytes];
    let mut bytes = 0;
    loop {
        let n = match source.read(&mut block) {
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if n == 0 {
            break;
        }
        bytes += n as u64;
    }
    Ok((bytes, source.finish().1))
}

pub(crate) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn check(role: &str, actual: &str, expected: &str) -> Result<()> {
    if actual != expected {
        return Err(Error::cache(format!("snapshot {role} SHA-256 mismatch"))
            .with_detail("role", role.to_owned()));
    }
    Ok(())
}

impl HashedFile {
    pub(crate) fn new(file: File) -> Self {
        Self {
            file,
            hash: Sha256::new(),
        }
    }
    pub(crate) fn finish(self) -> (File, String) {
        (self.file, format!("{:x}", self.hash.finalize()))
    }
}

impl Read for HashedFile {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let n = self.file.read(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }
}

impl Write for HashedFile {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let n = self.file.write(bytes)?;
        self.hash.update(&bytes[..n]);
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}
