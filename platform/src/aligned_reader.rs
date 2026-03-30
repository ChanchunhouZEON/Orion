/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::thread::ThreadId;

use crate::io_context::UnixIOContext;
use crossbeam::sync::ShardedLock;
use memmap2::Mmap;
use once_cell::sync::Lazy;

/// Disk I/O alignment constant (matches sector size).
pub const DISK_IO_ALIGNMENT: usize = 512;

/// Unix aligned file reader replacing Windows WindowsAlignedFileReader.
/// Uses memmap2 for memory-mapped file access instead of IOCP + OVERLAPPED batch reads.
/// Provides per-thread I/O contexts via ShardedLock<HashMap<ThreadId, ...>>.
pub struct UnixAlignedFileReader {
    file: File,
    mmap: Arc<Mmap>,
    file_size: u64,
    // ctx_map is the mapping from thread id to io context. It is hashmap behind a sharded lock to allow concurrent access from multiple threads.
    // ShardedLock: shardedlock provides an implementation of a reader-writer lock that offers concurrent read access to the shared data while allowing exclusive write access.
    // It achieves better scalability by dividing the shared data into multiple shards, and each with its own internal lock.
    // Multiple threads can read from different shards simultaneously, reducing contention.
    // https://docs.rs/crossbeam/0.8.2/crossbeam/sync/struct.ShardedLock.html
    // Comparing to RwLock, ShardedLock provides higher concurrency for read operations and is suitable for read heavy workloads.
    // The value of the hashmap is an Arc<IOContext> to allow immutable access to IOContext with automatic reference counting.
    contexts: Lazy<ShardedLock<HashMap<ThreadId, Arc<UnixIOContext>>>>,
}

impl UnixAlignedFileReader {
    /// Open a file for aligned reading via mmap.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = File::open(path)?;
        let file_size = file.metadata()?.len();
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(UnixAlignedFileReader {
            file,
            mmap: Arc::new(mmap),
            file_size,
            contexts: Lazy::new(|| ShardedLock::new(HashMap::new())),
        })
    }

    /// Get the file size.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Get the raw file descriptor.
    pub fn raw_fd(&self) -> i32 {
        self.file.as_raw_fd()
    }

    /// Read data from the file at the specified offset into the provided buffer.
    /// This copies from the mmap region into the aligned buffer (same API as Windows version).
    pub fn read(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let offset = offset as usize;
        let len = buf.len();

        if offset + len > self.file_size as usize {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Read past end of file: offset={}, len={}, file_size={}",
                    offset, len, self.file_size
                ),
            ));
        }

        buf.copy_from_slice(&self.mmap[offset..offset + len]);
        Ok(len)
    }

    /// Read data at the specified offset, returning a slice view into the mmap.
    /// Zero-copy access when alignment is not required.
    pub fn read_slice(&self, offset: u64, len: usize) -> io::Result<&[u8]> {
        let offset = offset as usize;

        if offset + len > self.file_size as usize {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "Read past end of file: offset={}, len={}, file_size={}",
                    offset, len, self.file_size
                ),
            ));
        }

        Ok(&self.mmap[offset..offset + len])
    }

    /// Get or create the I/O context for the current thread.
    pub fn get_ctx(&self) -> Arc<UnixIOContext> {
        let tid = std::thread::current().id();

        // Try read lock first
        if let Ok(guard) = self.contexts.read() {
            if let Some(ctx) = guard.get(&tid) {
                return Arc::clone(ctx);
            }
        }

        // Need to create - take write lock
        let mut guard = self.contexts.write().unwrap_or_else(|e| e.into_inner());
        let ctx = Arc::new(UnixIOContext::new(self.file.as_raw_fd()));
        guard.insert(tid, Arc::clone(&ctx));
        ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_aligned_reader_roundtrip() {
        let dir = std::env::temp_dir().join("diskann_test_aligned_reader");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("test.bin");

        // Write test data
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(&data).unwrap();
        }

        // Read via aligned reader
        let reader = UnixAlignedFileReader::open(&path).unwrap();
        assert_eq!(reader.file_size(), 1024);

        let mut buf = vec![0u8; 512];
        let n = reader.read(0, &mut buf).unwrap();
        assert_eq!(n, 512);
        assert_eq!(&buf[..], &data[..512]);

        let n = reader.read(512, &mut buf).unwrap();
        assert_eq!(n, 512);
        assert_eq!(&buf[..], &data[512..]);

        // Zero-copy slice access
        let slice = reader.read_slice(100, 50).unwrap();
        assert_eq!(slice, &data[100..150]);

        // Cleanup
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
