/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Immutable content-addressed objects and bounded metadata reads.
mod atomic;
mod lease;
use crate::integrity::{check, sha256, HashedFile};
use crate::{Error, Result};
pub(crate) use atomic::{publish, sync_dir, PendingFile};
pub(crate) use lease::WriterLease;
use orion_ann_platform::buffered_io::FILE_WRITE_BYTES;
use std::{
    fs,
    io::{self, BufWriter, Read, Write},
    path::{Path, PathBuf},
};
pub(crate) struct EncodedFile {
    pub temporary: PendingFile,
    pub bytes: u64,
    pub digest: String,
}

pub(crate) fn encode_file(
    path: &Path,
    encode: impl FnOnce(&mut BufWriter<HashedFile>) -> Result<()>,
) -> Result<EncodedFile> {
    let (temporary, file) = PendingFile::create(path)?;
    let mut writer = BufWriter::with_capacity(FILE_WRITE_BYTES, HashedFile::new(file));
    encode(&mut writer)?;
    writer.flush()?;
    let sink = writer.into_inner().map_err(|e| e.into_error())?;
    let (file, digest) = sink.finish();
    file.sync_all()?;
    Ok(EncodedFile {
        temporary,
        bytes: file.metadata()?.len(),
        digest,
    })
}

/// A sibling of the working directory so removing lookup records never unlinks
/// objects referenced by a snapshot. Capture an absolute path for later opens.
pub(crate) fn object_dir(cache: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(cache)?;
    let mut name = absolute
        .file_name()
        .ok_or_else(|| Error::invalid_argument("cache directory must have a name"))?
        .to_owned();
    name.push(".objects");
    Ok(absolute.with_file_name(name))
}

/// Publish a fully synced inode without clobbering a concurrent publisher.
pub(crate) fn put(
    directory: &Path,
    extension: &str,
    encode: impl FnOnce(&mut BufWriter<HashedFile>) -> Result<()>,
) -> Result<(u64, String)> {
    let mut dir = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        dir.mode(0o700);
    }
    dir.recursive(true).create(directory)?;
    let data = encode_file(&directory.join("pending"), encode)?;
    let path = object(directory, &data.digest, extension)?;
    match fs::hard_link(&data.temporary.0, &path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
            // Another writer or an earlier run published this digest. A corrupt
            // object may be repaired only with verified, byte-identical content.
            if crate::integrity::file_sha256(&path)? != data.digest {
                fs::rename(&data.temporary.0, &path)?;
            }
        }
        Err(e) => return Err(Error::io("publish object", &path, e)),
    }
    sync_dir(directory)?;
    if let Some(parent) = directory.parent() {
        sync_dir(parent)?;
    }
    Ok((data.bytes, data.digest))
}
pub(crate) fn read_record(
    dir: &Path,
    digest: &str,
    ext: &str,
    limit: u64,
    role: &str,
) -> Result<Vec<u8>> {
    let bytes = read_bounded(&object(dir, digest, ext)?, limit)?;
    check(role, &sha256(&bytes), digest)?;
    Ok(bytes)
}

pub(crate) fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let file = fs::File::open(path).map_err(|e| Error::io("open record", path, e))?;
    read_bounded_file(file, path, limit)
}

pub(crate) fn read_bounded_file(file: fs::File, path: &Path, limit: u64) -> Result<Vec<u8>> {
    let mut data = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut data)
        .map_err(|e| Error::io("read record", path, e))?;
    if data.len() as u64 > limit {
        return Err(Error::cache("oversized snapshot record")
            .with_detail("path", path.display().to_string()));
    }
    Ok(data)
}

pub(crate) fn object(dir: &Path, digest: &str, extension: &str) -> Result<PathBuf> {
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(Error::cache("invalid SHA-256 object name"));
    }
    Ok(dir.join(format!("{digest}.{extension}")))
}
