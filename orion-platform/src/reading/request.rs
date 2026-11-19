/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::{io, ops::Range};

/// A checked byte request relative to one immutable object's start.
///
/// Empty reads are permitted at any offset up to the object end. Disk block
/// rounding belongs to the backend; returned leases exclude rounded padding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadRequest {
    offset: u64,
    bytes: usize,
}

impl ReadRequest {
    /// Describe a logical range before selecting a reader or acquiring storage.
    ///
    /// `offset` is relative to the object start, and `bytes` is the exact payload
    /// length the consumer needs. This checks representability only; the reader
    /// checks the range against its object size through [`Self::validate`].
    ///
    /// # Examples
    /// ```
    /// use orion_ann_platform::reading::ReadRequest;
    /// let record = ReadRequest::new(16 + 7 * 384, 384)?;
    /// record.validate(16 + 100 * 384)?;
    /// # Ok::<(), std::io::Error>(())
    /// ```
    ///
    /// # Errors
    /// Rejects overflowing offsets and ranges too large for a Rust slice.
    #[inline]
    pub fn new(offset: u64, bytes: usize) -> io::Result<Self> {
        if bytes > isize::MAX as usize || offset.checked_add(bytes as u64).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read range overflow",
            ));
        }
        Ok(Self { offset, bytes })
    }

    /// Byte position in the coordinate system chosen by the consumer.
    /// Readers interpret it relative to the object; [`super::ReadLease::section`]
    /// interprets it relative to the current lease instead.
    pub fn offset(self) -> u64 {
        self.offset
    }

    /// Requested payload length, excluding any backend page rounding.
    pub fn byte_len(self) -> usize {
        self.bytes
    }

    /// Check whether this request fits a captured logical object length.
    /// An empty request at exactly `object_bytes` is valid; one beyond it is not.
    /// This does not inspect physical allocation or confirm that a file stayed
    /// unchanged after its length was captured.
    ///
    /// # Errors
    /// Rejects any range beyond the reader's captured logical size.
    #[inline]
    pub fn validate(self, object_bytes: u64) -> io::Result<()> {
        if self.offset + self.bytes as u64 > object_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "read range exceeds object length",
            ));
        }
        Ok(())
    }

    /// Validates the request against an addressable object and converts it to a slice range.
    pub(super) fn range(self, object_bytes: usize) -> io::Result<Range<usize>> {
        self.validate(object_bytes as u64)?;
        // Validation against a usize-sized object also proves the conversion.
        let start = self.offset as usize;
        Ok(start..start + self.bytes)
    }
}
