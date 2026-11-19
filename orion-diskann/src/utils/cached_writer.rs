/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use crate::common::ANNResult;
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

/// Sequential buffered writer with explicit flush and best-effort drop cleanup.
pub struct CachedWriter {
    writer: File,
    cache_size: u64,
    cache_buf: Vec<u8>,
    cur_off: u64,
    fsize: u64,
}

impl CachedWriter {
    /// Opens a file without truncating it and allocates a nonempty write cache.
    pub fn new(filename: &str, cache_size: u64) -> ANNResult<Self> {
        let writer = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(Path::new(filename))?;

        if cache_size == 0 {
            return Err(std::io::Error::other("Cache size must be greater than 0").into());
        }

        Ok(Self {
            writer,
            cache_size,
            cache_buf: vec![0; cache_size as usize],
            cur_off: 0,
            fsize: 0,
        })
    }

    /// Writes pending cached bytes and flushes the file handle.
    ///
    /// Call explicitly to observe write errors; Drop cannot report them. This is not fsync.
    pub fn flush(&mut self) -> ANNResult<()> {
        if self.cur_off > 0 {
            self.flush_cache()?;
        }
        self.writer.flush()?;
        Ok(())
    }

    /// Returns bytes already written through this writer, excluding pending cache bytes.
    pub fn get_file_size(&self) -> u64 {
        self.fsize
    }

    /// Stages small writes and sends oversized writes directly after draining pending bytes.
    pub fn write(&mut self, write_buf: &[u8]) -> ANNResult<()> {
        let n_bytes = write_buf.len() as u64;
        if n_bytes <= (self.cache_size - self.cur_off) {
            self.cache_buf[(self.cur_off as usize)..((self.cur_off + n_bytes) as usize)]
                .copy_from_slice(&write_buf[..n_bytes as usize]);
            self.cur_off += n_bytes;
        } else {
            self.writer
                .write_all(&self.cache_buf[..self.cur_off as usize])?;
            self.fsize += self.cur_off;
            self.writer.write_all(write_buf)?;
            self.fsize += n_bytes;
            self.cache_buf.fill(0);
            self.cur_off = 0;
        }
        Ok(())
    }

    /// Flushes pending bytes and seeks to the beginning without truncating the file.
    pub fn reset(&mut self) -> ANNResult<()> {
        self.flush_cache()?;
        self.writer.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    /// Writes the active cache prefix and resets its cursor for reuse.
    fn flush_cache(&mut self) -> ANNResult<()> {
        self.writer
            .write_all(&self.cache_buf[..self.cur_off as usize])?;
        self.fsize += self.cur_off;
        self.cache_buf.fill(0);
        self.cur_off = 0;
        Ok(())
    }
}

impl Drop for CachedWriter {
    /// Attempts a final flush while discarding errors that cannot be returned from Drop.
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
