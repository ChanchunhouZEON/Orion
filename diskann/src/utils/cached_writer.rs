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

pub struct CachedWriter {
    writer: File,
    cache_size: u64,
    cache_buf: Vec<u8>,
    cur_off: u64,
    fsize: u64,
}

impl CachedWriter {
    pub fn new(filename: &str, cache_size: u64) -> ANNResult<Self> {
        let writer = OpenOptions::new()
            .write(true)
            .create(true)
            .open(Path::new(filename))?;

        if cache_size == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Cache size must be greater than 0",
            )
            .into());
        }

        Ok(Self {
            writer,
            cache_size,
            cache_buf: vec![0; cache_size as usize],
            cur_off: 0,
            fsize: 0,
        })
    }

    pub fn flush(&mut self) -> ANNResult<()> {
        if self.cur_off > 0 {
            self.flush_cache()?;
        }
        self.writer.flush()?;
        Ok(())
    }

    pub fn get_file_size(&self) -> u64 {
        self.fsize
    }

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

    pub fn reset(&mut self) -> ANNResult<()> {
        self.flush_cache()?;
        self.writer.seek(SeekFrom::Start(0))?;
        Ok(())
    }

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
    fn drop(&mut self) {
        let _ = self.flush();
    }
}
