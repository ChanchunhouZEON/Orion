/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use orion_ann_platform::{
    allocation_budget::AllocationAccount,
    reading::{ObjectReader, ReadLease, ReadRequest},
    region::{MutableRegion, RegionOptions},
};
use std::{fs::File, future::Future, io, path::Path, pin::Pin, sync::Arc};
use tokio::sync::Semaphore;

/// Per-reader bounds. Clones of a reader share its in-flight limit.
/// Completed leases retain their allocation charges until their final drop;
/// the in-flight limit bounds operations, not completed results held by callers.
#[derive(Debug, Clone)]
pub struct FileReaderOptions {
    /// Maximum nonempty operations submitted to the blocking pool at once.
    /// Must be nonzero. Excess callers wait asynchronously before a buffer is
    /// allocated; all clones share this bound. Completed leases do not consume
    /// operation slots, even while they continue to retain their buffers.
    pub max_in_flight: usize,

    /// Maximum bytes returned by one request, not an aggregate memory budget.
    /// Requests exceeding this nonzero limit fail instead of being split.
    /// When wrapped by a cache, this must accommodate that cache's page size.
    pub max_read_bytes: usize,

    /// Minimum alignment of the owned output buffer, as a nonzero power of two.
    /// This controls typed/SIMD access to the result; it neither aligns file
    /// offsets nor enables direct I/O. The default is 4096 bytes.
    pub buffer_alignment: usize,

    /// Optional shared budget charged by each actual read-buffer allocation.
    /// Charges survive operation completion while leases retain the buffer.
    /// `None` leaves allocation unaccounted here; the OS may still impose limits.
    pub accounting: Option<AllocationAccount>,
}

impl Default for FileReaderOptions {
    /// Selects bounded read concurrency, per-operation bytes and buffer alignment.
    fn default() -> Self {
        Self {
            max_in_flight: 8,
            max_read_bytes: 8 * 1024 * 1024,
            buffer_alignment: 4096,
            accounting: None,
        }
    }
}

/// Read immutable generation files without blocking an async executor worker.
///
/// [`ObjectReader::read`] submits buffered positional I/O to Tokio's blocking
/// pool. Concurrent reads do not share a seek cursor. This is the portable
/// baseline backend, not io_uring, direct I/O or a kernel-bypass implementation.
///
/// The owner must keep the opened file contents unchanged. Length is captured
/// at open time; same-length overwrites are not detected. Successful leases own
/// copied bytes and can outlive this reader and its runtime.
///
/// # Cancellation
/// Dropping a read future stops waiting for its result. An operation already
/// submitted may continue, retaining its buffer, file and slot until it finishes.
/// Completed leases keep their buffer accounting, but no longer hold an I/O slot.
#[derive(Debug, Clone)]
pub struct FileReader {
    file: Arc<File>,
    bytes: u64,
    permits: Arc<Semaphore>,
    options: FileReaderOptions,
}

impl FileReader {
    /// Bind this reader to an existing regular file and capture its byte length.
    ///
    /// Call during preparation: opening and metadata lookup are synchronous and
    /// need no Tokio context. Polling a nonempty read later requires a runtime
    /// with blocking workers. No file authentication or eager payload read occurs.
    ///
    /// # Errors
    /// Reports open/metadata failures, non-regular files, zero or unrepresentable
    /// bounds, and invalid buffer alignment.
    pub fn open(path: impl AsRef<Path>, options: FileReaderOptions) -> io::Result<Self> {
        if options.max_in_flight == 0
            || options.max_in_flight > Semaphore::MAX_PERMITS
            || options.max_read_bytes == 0
            || options.max_read_bytes > isize::MAX as usize
            || std::alloc::Layout::from_size_align(options.max_read_bytes, options.buffer_alignment)
                .is_err()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid file reader limits or alignment",
            ));
        }
        let file = File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "object reader requires a regular file",
            ));
        }
        Ok(Self {
            bytes: metadata.len(),
            file: Arc::new(file),
            permits: Arc::new(Semaphore::new(options.max_in_flight)),
            options,
        })
    }

    // Keep the blocking operation injectable privately so cancellation tests can
    // stop it at an exact boundary without timing assumptions or slow devices.
    /// Reserves a bounded read slot and owns its buffer through completion, including cancellation.
    async fn read_with(
        &self,
        request: ReadRequest,
        operation: impl FnOnce(&File, &mut [u8], u64) -> io::Result<()> + Send + 'static,
    ) -> io::Result<ReadLease<'static>> {
        request.validate(self.bytes)?;
        if request.byte_len() > self.options.max_read_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read exceeds per-operation byte limit",
            ));
        }
        let storage = RegionOptions {
            accounting: self.options.accounting.clone(),
            ..Default::default()
        };
        // Empty requests neither acquire a queue slot nor require a runtime.
        if request.byte_len() == 0 {
            return Ok(ReadLease::from_region(
                MutableRegion::new(0, self.options.buffer_alignment, &storage)?.freeze()?,
            ));
        }
        let runtime = tokio::runtime::Handle::try_current().map_err(|_| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "file reads require a Tokio runtime",
            )
        })?;
        let permit = self
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| io::Error::other("file reader queue closed"))?;
        let file = self.file.clone();
        let alignment = self.options.buffer_alignment;

        // Ownership moves INTO the blocking operation, not the awaiting task.
        // Abandoning its JoinHandle cannot release a buffer or queue slot still
        // in use by the OS. Errors and unwinds release both through RAII.
        runtime
            .spawn_blocking(move || {
                let _permit = permit;
                let mut buffer = MutableRegion::new(request.byte_len(), alignment, &storage)?;
                operation(&file, buffer.as_bytes_mut(), request.offset())?;
                Ok(ReadLease::from_region(buffer.freeze()?))
            })
            .await
            .map_err(|error| io::Error::other(format!("file read worker failed: {error}")))?
    }
}

impl ObjectReader for FileReader {
    // Only explicit I/O pays for a boxed suspension state. Resident readers use
    // std::future::Ready, and no Tokio dependency reaches their implementation.
    type Read<'a> = Pin<Box<dyn Future<Output = io::Result<ReadLease<'a>>> + Send + 'a>>;

    /// Returns the file length captured when this reader was opened.
    fn byte_len(&self) -> u64 {
        self.bytes
    }

    /// Schedules a bounded positioned read and returns an owning lease for its completed bytes.
    fn read(&self, request: ReadRequest) -> Self::Read<'_> {
        Box::pin(async move { self.read_with(request, read_exact_at).await })
    }
}

/// Fills one buffer with positioned reads, retrying interruptions and rejecting premature EOF.
fn read_exact_at(file: &File, mut output: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !output.is_empty() {
        #[cfg(unix)]
        let result = std::os::unix::fs::FileExt::read_at(file, output, offset);
        #[cfg(windows)]
        let result = std::os::windows::fs::FileExt::seek_read(file, output, offset);
        #[cfg(not(any(unix, windows)))]
        let result: io::Result<usize> = Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "positioned reads unavailable",
        ));
        match result {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "object shortened during read",
                ))
            }
            Ok(n) => {
                offset += n as u64;
                output = &mut output[n..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
