/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Runtime-independent acquisition of immutable byte ranges.
//!
//! A reader is bound to one object; offsets are relative to that object, not
//! graph node IDs. Layout adapters resolve logical records before requesting
//! bytes. Successful reads return exactly the requested bytes or an error.
//! They do not authenticate a file or interpret its format.
//!
//! Resident reads complete immediately without an executor, allocation, copy,
//! or reference-count increment. Mapped reads also return a view immediately,
//! but dereferencing that view can still cause a blocking OS page fault.
//! Explicit file readers may suspend; their buffers must outlive pending I/O.

mod lease;
mod request;

pub use lease::ReadLease;
pub use request::ReadRequest;

use crate::region::ReadRegion;
use std::{
    future::{ready, Future, Ready},
    io,
};

/// Acquires immutable data without exposing an executor to its consumers.
///
/// This trait uses concrete associated futures so resident readers need no
/// boxed future or task dispatch. Computation borrows the completed lease;
/// callers should acquire records/batches, not individual SIMD chunks.
///
/// A reader describes one immutable generation. It does not decide whether the
/// bytes represent a graph, vectors or a sidecar, and never performs metric or
/// row-layout conversion. The owner must prevent mutation of that generation
/// for the reader's lifetime; format and content checks belong to its caller.
pub trait ObjectReader: Send + Sync {
    /// Acquisition state borrowing this reader until completion or cancellation.
    /// Immediate backends may use [`Ready`]; I/O backends can retain operation
    /// state across polls. A completed lease supplies the lifetime of the bytes.
    type Read<'a>: Future<Output = io::Result<ReadLease<'a>>> + Send + 'a
    where
        Self: 'a;

    /// Logical size captured by this reader, independent of cache residency.
    fn byte_len(&self) -> u64;

    /// Acquire exactly the requested range. Dropping the future abandons the
    /// result; it does not necessarily cancel an underlying operating-system read.
    ///
    /// # Errors
    /// Rejects invalid ranges; file backends additionally report short reads,
    /// I/O failures, allocation limits, and missing executor context.
    fn read(&self, request: ReadRequest) -> Self::Read<'_>;
}

impl ObjectReader for ReadRegion {
    type Read<'a> = Ready<io::Result<ReadLease<'a>>>;

    /// Returns the immutable region span used to validate read requests.
    #[inline]
    fn byte_len(&self) -> u64 {
        self.len_bytes() as u64
    }

    /// Returns an immediately ready lease without copying; mapped bytes may still fault during access.
    #[inline]
    fn read(&self, request: ReadRequest) -> Self::Read<'_> {
        ready(ReadLease::borrow(self, request))
    }
}
