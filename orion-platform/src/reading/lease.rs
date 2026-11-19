/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use super::ReadRequest;
use crate::region::ReadRegion;
use bytemuck::Pod;
use std::{fmt::Debug, io, ops::Range, sync::Arc};

/// Type-erased owner retained until the last borrowed read lease is released.
trait Retention: Send + Sync + Debug {}
impl<T: Send + Sync + Debug> Retention for T {}

/// Immutable bytes whose backing remains alive throughout their use.
///
/// A borrowed lease does not increment a backing's reference count. Explicit
/// promotion to ownership shares its allocation/accounting, without copying.
/// Owning a lease prevents backing reclamation; it does not imply OS mlock.
///
/// ```compile_fail
/// use orion_ann_platform::{reading::{ObjectReader, ReadRequest}, region::ArrayView};
/// let rows = ArrayView::copy_from(&[1u32, 2], &Default::default()).unwrap();
/// let pending = rows.region().read(ReadRequest::new(0, 8).unwrap());
/// drop(rows);
/// drop(pending);
/// ```
#[derive(Debug)]
pub struct ReadLease<'a> {
    backing: LeaseBacking<'a>,
    // Drop backing before returning external resources such as cache capacity.
    retention: Option<Arc<dyn Retention>>,
}

/// Borrowed or shared backing that keeps a readable byte span alive.
#[derive(Debug)]
enum LeaseBacking<'a> {
    Borrowed {
        region: &'a ReadRegion,
        range: Range<usize>,
    },
    Owned(ReadRegion),
}

impl<'a> ReadLease<'a> {
    /// Creates a range-checked lease retaining a borrow of an immutable resident or mapped region.
    pub(super) fn borrow(region: &'a ReadRegion, request: ReadRequest) -> io::Result<Self> {
        Ok(Self {
            backing: LeaseBacking::Borrowed {
                region,
                range: request.range(region.len_bytes())?,
            },
            retention: None,
        })
    }

    /// Borrow exactly the acquired payload while retaining its backing owner.
    /// No read or copy is initiated here. Access to mapped bytes may still fault
    /// pages in; this borrow is not a promise that the payload is resident RAM.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        match &self.backing {
            LeaseBacking::Borrowed { region, range } => &region.as_bytes()[range.clone()],
            LeaseBacking::Owned(region) => region.as_bytes(),
        }
    }

    /// Checked POD interpretation. Logical shape and SIMD padding remain the
    /// format adapter's responsibility. No copy or allocation is performed.
    /// The cast uses native byte order; it does not decode a wire format.
    ///
    /// # Errors
    /// Rejects zero-sized elements, insufficient address alignment and lengths
    /// that cannot be represented as a whole number of `T` elements.
    #[inline]
    pub fn as_slice<T: Pod>(&self) -> io::Result<&[T]> {
        if size_of::<T>() == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "zero-sized read element",
            ));
        }
        bytemuck::try_cast_slice(self.as_bytes()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "read bytes do not match element layout",
            )
        })
    }

    /// Keep a subrange after this lease or its reader has been dropped.
    ///
    /// The request is relative to this lease, not the original object. The
    /// returned owned view shares backing and retention guards: selecting a few
    /// bytes can therefore keep an entire cache page and its capacity alive.
    ///
    /// # Errors
    /// Rejects ranges outside this lease's payload.
    pub fn section(&self, request: ReadRequest) -> io::Result<ReadLease<'static>> {
        let sub = request.range(self.as_bytes().len())?;
        let region = match &self.backing {
            LeaseBacking::Borrowed { region, range } => {
                region.section(range.start + sub.start..range.start + sub.end)?
            }
            LeaseBacking::Owned(region) => region.section(sub)?,
        };
        Ok(ReadLease {
            backing: LeaseBacking::Owned(region),
            retention: self.retention.clone(),
        })
    }

    /// Move a result into longer-lived work without borrowing the reader.
    ///
    /// Owned leases transfer their backing unchanged; borrowed leases acquire
    /// shared ownership. Neither path copies bytes or reserves a second payload.
    /// Drop the returned lease when processing is done to release its share of
    /// the backing and any attached capacity guards.
    ///
    /// # Errors
    /// Propagates backing-range validation when promoting a borrowed view.
    pub fn into_owned(self) -> io::Result<ReadLease<'static>> {
        let region = match self.backing {
            LeaseBacking::Borrowed { region, range } => region.section(range)?,
            LeaseBacking::Owned(region) => region,
        };
        Ok(ReadLease {
            backing: LeaseBacking::Owned(region),
            retention: self.retention,
        })
    }

    /// Tie a resource guard to the returned data, for composing reader adapters.
    ///
    /// A page cache uses this to keep a frame reservation alive while callers
    /// consume its bytes. The owner cannot replace the payload. Existing guards
    /// remain retained, including through [`Self::section`] and
    /// [`Self::into_owned`], so wrapping a reader cannot release upstream
    /// capacity early. The last shared guard owner determines reclamation.
    pub fn retain<T: Send + Sync + Debug + 'static>(mut self, owner: Arc<T>) -> Self {
        self.retention = Some(match self.retention.take() {
            None => owner,
            Some(previous) => Arc::new((previous, owner)),
        });
        self
    }
}

impl ReadLease<'static> {
    /// Publish a backend's completed immutable buffer as a read result.
    /// Ownership of the region and its allocation accounting moves into the
    /// lease. This is the handoff after I/O completes, not a way to expose a
    /// buffer while another task is still writing it.
    pub fn from_region(region: ReadRegion) -> Self {
        Self {
            backing: LeaseBacking::Owned(region),
            retention: None,
        }
    }
}
