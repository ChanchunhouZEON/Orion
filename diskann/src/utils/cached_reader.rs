/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::fs::File;
use std::io::{Read, Seek};

use crate::common::{ANNError, ANNResult};

pub struct CachedReader {
    reader: File,
    cache_size: u64,
    cache_buf: Vec<u8>,
    cur_off: u64,
    fsize: u64,
}

impl CachedReader {
    pub fn new(filename: &str, cache_size: u64) -> ANNResult<Self> {
        let mut reader = File::open(filename)?;
        let metadata = reader.metadata()?;
        let fsize = metadata.len();

        let cache_size = cache_size.min(fsize);
        let mut cache_buf = vec![0; cache_size as usize];
        reader.read_exact(&mut cache_buf)?;

        Ok(Self {
            reader,
            cache_size,
            cache_buf,
            cur_off: 0,
            fsize,
        })
    }

    pub fn get_file_size(&self) -> u64 {
        self.fsize
    }

    pub fn read(&mut self, read_buf: &mut [u8]) -> ANNResult<()> {
        let n_bytes = read_buf.len() as u64;
        if n_bytes <= (self.cache_size - self.cur_off) {
            read_buf.copy_from_slice(
                &self.cache_buf
                    [(self.cur_off as usize)..(self.cur_off as usize + n_bytes as usize)],
            );
            self.cur_off += n_bytes;
        } else {
            let cached_bytes = self.cache_size - self.cur_off;
            if n_bytes - cached_bytes > self.fsize - self.reader.stream_position()? {
                return Err(ANNError::log_index_error(format!(
                    "Reading beyond end of file, n_bytes: {} cached_bytes: {} fsize: {} current pos: {}",
                    n_bytes,
                    cached_bytes,
                    self.fsize,
                    self.reader.stream_position()?
                )));
            }

            read_buf[..cached_bytes as usize]
                .copy_from_slice(&self.cache_buf[self.cur_off as usize..]);
            self.reader
                .read_exact(&mut read_buf[cached_bytes as usize..])?;
            self.cur_off = self.cache_size;

            let size_left = self.fsize - self.reader.stream_position()?;
            if size_left >= self.cache_size {
                self.reader.read_exact(&mut self.cache_buf)?;
                self.cur_off = 0;
            }
        }
        Ok(())
    }

    pub fn read_u32(&mut self) -> ANNResult<u32> {
        let mut bytes = [0u8; 4];
        self.read(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
}
