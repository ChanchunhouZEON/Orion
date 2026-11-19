/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Immutable storage ownership, independent of object formats and search semantics.
//! A view owns its backing through Arc. Construction is exclusive; publishing a
//! view removes all writable access. File mappings only cover private temporary
//! files: callers cannot invalidate a slice by overwriting an external cache.
use crate::allocation_budget::{AllocationAccount, AllocationLease};
use bytemuck::Pod;
use std::{
    alloc::{dealloc, Layout},
    io,
    marker::PhantomData,
    ops::Range,
    path::PathBuf,
    ptr::NonNull,
    sync::Arc,
};

/// Physical placement only; locking and cache advice are separate policies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Placement {
    /// Own an allocation in the process heap. This does not implicitly lock
    /// pages into RAM; residency and swap remain separate policies.
    #[default]
    Resident,
    /// Own a private file mapping whose pages the OS may reclaim and reload.
    /// Mapped bytes remain directly addressable; dereferencing can block on I/O.
    Mapped,
}

/// Allocation policy supplied before constructing a region or typed array.
/// Choosing mapped storage changes backing ownership, not the consumer's slice
/// layout. It does not opt a search session into asynchronous I/O.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegionOptions {
    /// Backing kind for the new allocation; see [`Placement`] for its guarantees.
    pub placement: Placement,

    /// Existing directory for private mapped backing files; `None` uses the OS
    /// temporary directory. Use a disk-backed filesystem to reduce memory
    /// pressure: a tmpfs backing still consumes memory. Ignored for resident data.
    pub directory: Option<PathBuf>,

    /// Optional domain/object account charged once per actual backing.
    /// Views and clones share that charge until the last owner drops. This tracks
    /// allocated bytes by placement, not live OS page residency.
    pub accounting: Option<AllocationAccount>,

    /// Request writeback and file-page reclamation at mapped construction checkpoints.
    /// Disabled by default to preserve ordinary warm-cache loading. Low-memory
    /// experiments enable it for every arm; it never runs in a distance kernel.
    /// The OS may retain or refault pages, so this does not promise an RSS bound.
    pub reclaim_after_write: bool,
}

/// Aligned heap owner used while a region is constructed and after it is frozen.
struct Allocation {
    ptr: NonNull<u8>,
    layout: Layout,
}

impl Allocation {
    /// Allocates and zero-initializes aligned storage, applying Linux huge-page advice before touching it.
    fn new(bytes: usize, alignment: usize) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        let alignment = if bytes >= crate::memory::HUGE_PAGE_ALIGNMENT {
            alignment.max(crate::memory::HUGE_PAGE_ALIGNMENT)
        } else {
            alignment
        };
        let layout = Layout::from_size_align(bytes.max(1), alignment)
            .map_err(|_| invalid("invalid region size or alignment"))?;
        // SAFETY: checked layout; ownership is retained until Drop. Advise
        // before touching large Linux allocations to preserve the THP policy.
        #[cfg(target_os = "linux")]
        let raw = unsafe { std::alloc::alloc(layout) };
        #[cfg(not(target_os = "linux"))]
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).ok_or_else(|| io::Error::other("region allocation failed"))?;
        #[cfg(target_os = "linux")]
        unsafe {
            let _ = crate::memory::advise_huge_pages(ptr.as_ptr(), bytes);
        }
        #[cfg(target_os = "linux")]
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, layout.size());
        }
        Ok(Self { ptr, layout })
    }

    /// Borrows an initialized construction prefix; callers keep its length within the owned layout.
    fn bytes_mut(&mut self, len: usize) -> &mut [u8] {
        // SAFETY: initialized allocation, exclusive borrow, bounded by constructor.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }
}

impl Drop for Allocation {
    /// Deallocates the aligned backing with the same layout used at construction.
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}
// SAFETY: allocation is uniquely writable during construction, then immutable.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

/// Exclusive construction storage. Mutation requires a unique borrow, and
/// private backing files never expose a pathname or writable file handle.
/// Unlike a copy-on-write mapping, dirty pages can be written back and reclaimed.
pub struct MutableRegion {
    ptr: NonNull<u8>,
    len: usize,
    backing: MutableBacking,
    // Field order frees/unmaps storage before returning its budget.
    lease: Option<AllocationLease>,
}

/// Exclusive heap or mapped bytes before immutable region publication.
enum MutableBacking {
    Owned(Allocation),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped {
        map: memmap2::MmapMut,
        file: std::fs::File,
    },
}
// SAFETY: byte access follows ordinary shared/exclusive borrowing. Backing is
// uniquely owned and no other handle can mutate or truncate its private inode.
unsafe impl Send for MutableRegion {}
unsafe impl Sync for MutableRegion {}
impl std::fmt::Debug for MutableRegion {
    /// Formats `bytes`, `placement` for diagnostic inspection.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutableRegion")
            .field("bytes", &self.len)
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}

impl MutableRegion {
    /// Reserve the final backing and expose exclusive zero-initialized bytes.
    ///
    /// Use for producers that fill a known-size buffer directly, then call
    /// [`Self::freeze`] to publish it. `alignment` must be a valid allocation
    /// alignment; mapped storage also requires an address the OS can satisfy.
    /// Empty regions use an owned allocation even when mapping was requested.
    ///
    /// # Errors
    /// Rejects invalid layouts and propagates budget, allocation, filesystem or
    /// mapping failures. A failed construction releases its reservation.
    pub fn new(bytes: usize, alignment: usize, options: &RegionOptions) -> io::Result<Self> {
        Layout::from_size_align(bytes, alignment).map_err(|_| invalid("invalid region layout"))?;
        let actual = if bytes == 0 {
            Placement::Resident
        } else {
            options.placement
        };
        let lease = options
            .accounting
            .as_ref()
            .map(|a| a.reserve(bytes, actual))
            .transpose()?;
        let backing = if options.placement == Placement::Resident || bytes == 0 {
            MutableBacking::Owned(Allocation::new(bytes, alignment)?)
        } else {
            #[cfg(target_arch = "wasm32")]
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "file mappings require a native filesystem",
            ));
            #[cfg(not(target_arch = "wasm32"))]
            {
                let mut file = match &options.directory {
                    Some(path) => tempfile::tempfile_in(path)?,
                    None => tempfile::tempfile()?,
                };
                reserve_backing(&mut file, bytes)?;
                // SAFETY: exact-length private zero-filled file; no writable
                // handle escapes. The mapping keeps its inode alive after close.
                let mapping = unsafe { memmap2::MmapMut::map_mut(&file)? };
                if !mapping.as_ptr().addr().is_multiple_of(alignment) {
                    return Err(invalid("mapping alignment unsupported"));
                }
                MutableBacking::Mapped { map: mapping, file }
            }
        };
        let ptr = match &backing {
            MutableBacking::Owned(a) => a.ptr,
            #[cfg(not(target_arch = "wasm32"))]
            MutableBacking::Mapped { map, .. } => NonNull::new(map.as_ptr().cast_mut())
                .ok_or_else(|| invalid("nonempty mapping has a null address"))?,
        };
        Ok(Self {
            ptr,
            len: bytes,
            backing,
            lease,
        })
    }

    /// Inspect initialized bytes while construction still owns the buffer.
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: initialized, owned bytes; cached length is validated on construction.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Fill the final buffer without an intermediate payload allocation.
    /// The exclusive borrow prevents publication or shared reads during writes.
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: unique borrow excludes every other reference to these bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Actual backing kind, including the owned fallback for an empty region.
    pub fn placement(&self) -> Placement {
        match self.backing {
            MutableBacking::Owned(_) => Placement::Resident,
            #[cfg(not(target_arch = "wasm32"))]
            MutableBacking::Mapped { .. } => Placement::Mapped,
        }
    }

    /// Persist a completed byte range and request release of its file-cache pages.
    /// Heap storage is unchanged. The mapping and allocation lease remain alive;
    /// later reads fault the original bytes back in. See `file_cache` for OS limits.
    ///
    /// # Errors
    /// Rejects invalid ranges and propagates writeback/advice failures. Requires
    /// Linux or macOS for nonempty mapped ranges; failures never unmap the owner.
    pub fn flush_and_reclaim(&mut self, range: Range<usize>) -> io::Result<()> {
        if range.start > range.end || range.end > self.len {
            return Err(invalid("region cache range out of bounds"));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let MutableBacking::Mapped { map, file } = &self.backing {
            // SAFETY: exclusive owner, shared private inode, checked byte range.
            unsafe {
                crate::file_cache::reclaim_mapping(map.as_ptr(), self.len, file, range)?;
            }
        }
        Ok(())
    }

    /// Adopt an existing uniquely owned buffer into a budget without copying it.
    /// Repeating the same adoption is harmless; changing accounting domains is rejected.
    /// Prefer construction-time accounting when the budget must constrain the
    /// allocation itself; adoption charges storage that already exists.
    ///
    /// # Errors
    /// Rejects a different existing account or an unavailable budget reservation.
    pub fn account_storage(&mut self, account: &AllocationAccount) -> io::Result<()> {
        if let Some(lease) = &self.lease {
            return if lease.account() == account {
                Ok(())
            } else {
                Err(invalid("buffer already belongs to another account"))
            };
        }
        self.lease = Some(account.reserve(self.len, self.placement())?);
        Ok(())
    }

    /// Publish this backing as immutable shared storage, without copying bytes.
    /// The allocation charge moves with it. For mapped storage this removes
    /// writable access; it is not a durable snapshot save or explicit flush API.
    ///
    /// # Errors
    /// Propagates failure to change a mapping to read-only protection.
    pub fn freeze(self) -> io::Result<ReadRegion> {
        let backing = match self.backing {
            MutableBacking::Owned(a) => Backing::Owned(a),
            #[cfg(not(target_arch = "wasm32"))]
            MutableBacking::Mapped { map, file } => Backing::Mapped {
                map: map.make_read_only()?,
                file,
            },
        };
        Ok(ReadRegion {
            backing: Arc::new(AccountedBacking {
                len: self.len,
                backing,
                lease: self.lease,
            }),
            offset: 0,
            len: self.len,
        })
    }
}

/// Avoid publishing a sparse writable mapping whose first store could fail with
/// SIGBUS on ENOSPC. Linux reserves extents; other filesystems use bounded writes
/// and sync to surface ordinary allocation errors before mapping.
#[cfg(not(target_arch = "wasm32"))]
fn reserve_backing(file: &mut std::fs::File, bytes: usize) -> io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: live private descriptor and representable positive file length.
        let code = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, bytes as libc::off_t) };
        if code == 0 {
            return file.set_len(bytes as u64);
        }
        if ![libc::EOPNOTSUPP, libc::ENOSYS, libc::EINVAL].contains(&code) {
            return Err(io::Error::from_raw_os_error(code));
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        let mut allocation = libc::fstore_t {
            fst_flags: libc::F_ALLOCATEALL,
            fst_posmode: libc::F_PEOFPOSMODE,
            fst_offset: 0,
            fst_length: bytes as libc::off_t,
            fst_bytesalloc: 0,
        };
        // SAFETY: private empty file and a fully initialized fstore_t. Reserving
        // physical extents does not publish old bytes; set_len zeroes the range.
        let result = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_PREALLOCATE, &mut allocation) };
        if result != -1 {
            return file.set_len(bytes as u64);
        }
        let error = io::Error::last_os_error();
        if !matches!(
            error.raw_os_error(),
            Some(libc::ENOTSUP | libc::EINVAL | libc::ENOSYS)
        ) {
            return Err(error);
        }
    }
    use std::io::Write;
    let block = vec![0u8; bytes.min(crate::buffered_io::FILE_WRITE_BYTES)];
    let mut remaining = bytes;
    while remaining != 0 {
        let n = remaining.min(block.len());
        file.write_all(&block[..n])?;
        remaining -= n;
    }
    file.sync_data()
}

/// Published heap or mapped storage shared by immutable region owners.
enum Backing {
    Owned(Allocation),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped {
        map: memmap2::Mmap,
        file: std::fs::File,
    },
}

impl Backing {
    /// Returns the stable backing address shared by immutable region views.
    fn ptr(&self) -> *const u8 {
        match self {
            Self::Owned(a) => a.ptr.as_ptr(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped { map, .. } => map.as_ptr(),
        }
    }

    /// Identifies heap versus file-backed ownership without inspecting page residency.
    fn placement(&self) -> Placement {
        match self {
            Self::Owned(_) => Placement::Resident,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped { .. } => Placement::Mapped,
        }
    }
}

// Every clone/section shares this owner, so the lease survives the parent view.
/// Backing storage paired with the allocation charge released after its bytes.
struct AccountedBacking {
    len: usize,
    backing: Backing,
    lease: Option<AllocationLease>,
}
/// An immutable view that keeps its entire backing and budget charge alive.
/// Clones and sections share ownership rather than copy bytes. A small section
/// can retain a much larger allocation; dropping the original view does not
/// reclaim that backing while any derived view or read lease still holds it.
#[derive(Clone)]
pub struct ReadRegion {
    backing: Arc<AccountedBacking>,
    offset: usize,
    len: usize,
}

impl std::fmt::Debug for ReadRegion {
    /// Formats `bytes`, `placement` for diagnostic inspection.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadRegion")
            .field("bytes", &self.len)
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}

impl ReadRegion {
    /// Logical byte length without touching mapped pages.
    #[inline]
    pub fn len_bytes(&self) -> usize {
        self.len
    }

    /// Borrow the view's payload without copying or acquiring another owner.
    /// Mapped bytes remain subject to ordinary OS page faults.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: constructors/range validate length, backing lives with the view.
        unsafe { std::slice::from_raw_parts(self.backing.backing.ptr().add(self.offset), self.len) }
    }

    /// Allocation kind shared by all views into this backing, not an RSS status.
    pub fn placement(&self) -> Placement {
        self.backing.backing.placement()
    }

    /// Request file-page reclamation without releasing this view or its shared lease.
    /// The backing is immutable, so other readers may safely refault the same bytes.
    /// Boundary pages can include adjacent views; resident storage is a no-op.
    ///
    /// # Errors
    /// Rejects ranges outside this view and propagates writeback/advice errors.
    pub fn reclaim_file_pages(&self, range: Range<usize>) -> io::Result<()> {
        if range.start > range.end || range.end > self.len {
            return Err(invalid("region cache range out of bounds"));
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Backing::Mapped { map, file } = &self.backing.backing {
            // SAFETY: immutable private backing remains owned by every view.
            unsafe {
                crate::file_cache::reclaim_mapping(
                    map.as_ptr(),
                    self.backing.len,
                    file,
                    self.offset + range.start..self.offset + range.end,
                )?;
            }
        }
        Ok(())
    }

    /// Return the original account for replacement allocations, without a new charge.
    pub fn accounting(&self) -> Option<AllocationAccount> {
        self.backing.lease.as_ref().map(|l| l.account().clone())
    }

    /// Account a freshly constructed, uniquely owned region (for example a fitted codebook).
    /// An already-accounted shared view can be checked without charging it again.
    /// Charges cover the whole original allocation, even if `self` is a section.
    ///
    /// # Errors
    /// Rejects adoption after unaccounted storage was shared, a conflicting
    /// existing account, or failure to reserve the full backing's budget.
    pub fn account_storage(&mut self, account: &AllocationAccount) -> io::Result<()> {
        if let Some(lease) = &self.backing.lease {
            return if lease.account() == account {
                Ok(())
            } else {
                Err(invalid("region already belongs to another account"))
            };
        }
        let owner = Arc::get_mut(&mut self.backing)
            .ok_or_else(|| invalid("cannot adopt an unaccounted shared region"))?;
        owner.lease = Some(account.reserve(owner.len, owner.backing.placement())?);
        Ok(())
    }

    /// Create an owned subview relative to this view's start, without copying.
    /// It remains valid independently of the parent; see [`ReadRegion`] for the
    /// lifetime and accounting consequences of retaining a small section.
    ///
    /// # Errors
    /// Rejects reversed ranges and ranges extending beyond this view.
    pub fn section(&self, range: Range<usize>) -> io::Result<Self> {
        if range.start > range.end || range.end > self.len {
            return Err(invalid("region range out of bounds"));
        }
        Ok(Self {
            backing: self.backing.clone(),
            offset: self.offset + range.start,
            len: range.len(),
        })
    }

    /// Attach a typed view for kernels that consume a contiguous POD slice.
    /// This shares ownership and uses native byte order; it does not decode,
    /// normalize or add SIMD row padding.
    ///
    /// # Errors
    /// Rejects zero-sized elements, a partial final element or insufficient
    /// alignment at this view's start (including an unaligned section offset).
    pub fn array<T: Pod>(&self) -> io::Result<ArrayView<T>> {
        let bytes = self.as_bytes();
        if size_of::<T>() == 0
            || !bytes.len().is_multiple_of(size_of::<T>())
            || !bytes.as_ptr().addr().is_multiple_of(align_of::<T>())
        {
            return Err(invalid("region does not match element layout"));
        }
        Ok(ArrayView {
            ptr: bytes.as_ptr().cast(),
            len: bytes.len() / size_of::<T>(),
            region: self.clone(),
            marker: PhantomData,
        })
    }
}

/// Stable, typed storage. Cached pointer/length keep backing dispatch out of SIMD.
#[derive(Clone)]
pub struct ArrayView<T: Pod> {
    ptr: *const T,
    len: usize,
    region: ReadRegion,
    marker: PhantomData<T>,
}
// SAFETY: POD view is immutable and owns the backing; no writable handles escape.
unsafe impl<T: Pod + Send + Sync> Send for ArrayView<T> {}
unsafe impl<T: Pod + Send + Sync> Sync for ArrayView<T> {}
impl<T: Pod> std::fmt::Debug for ArrayView<T> {
    /// Formats the diagnostic representation of this wrapper and its contained state.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.region.fmt(f)
    }
}

impl<T: Pod> std::ops::Deref for ArrayView<T> {
    type Target = [T];

    /// Borrows the checked typed slice without copying or changing backing ownership.
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Pod> ArrayView<T> {
    /// Copy a borrowed payload into independently owned storage.
    /// Use [`ReadRegion::array`] instead when a compatible region already owns
    /// the bytes. This operation treats values as a flat array, not padded rows.
    ///
    /// # Errors
    /// Propagates the layout, budget and backing failures of [`Self::build_rows`].
    pub fn copy_from(values: &[T], options: &RegionOptions) -> io::Result<Self> {
        Self::build_rows(values.len(), 1, 64, options, |start, out| {
            out.copy_from_slice(&values[start..start + out.len()]);
            Ok(())
        })
    }

    /// Borrow the whole initialized array, including any producer-supplied padding.
    /// Access preserves the cached pointer; it performs no backing dispatch per
    /// element, but mapped pages can still fault into memory.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the checked view owns immutable, aligned POD storage. Its
        // cached pointer and length cannot change independently of that owner.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Stable start address for low-level consumers while an owner stays alive.
    /// This grants no writable access and does not extend the backing's lifetime.
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Borrow the backing view for byte-range reads or allocation inspection.
    pub fn region(&self) -> &ReadRegion {
        &self.region
    }

    /// Allocation kind of the shared backing; see [`Placement`].
    pub fn placement(&self) -> Placement {
        self.region.placement()
    }

    /// Adopt this array using [`ReadRegion::account_storage`].
    /// The same uniqueness, whole-backing charge and error rules apply.
    pub fn account_storage(&mut self, account: &AllocationAccount) -> io::Result<()> {
        self.region.account_storage(account)
    }

    /// Build row-oriented storage incrementally, then publish it as immutable.
    ///
    /// `stride` is the number of `T` elements per physical row, including padding;
    /// `alignment` is in bytes and is raised to at least `align_of::<T>()`.
    /// `fill` receives the first row index and a zero-initialized slice containing
    /// complete rows. It writes logical values and leaves unused padding zero.
    ///
    /// Resident placement calls `fill` once on the final allocation. Mapped
    /// placement reuses a bounded block and may call it many times before mapping
    /// the private file read-only. A callback must not depend on chunk boundaries.
    /// Temporary construction buffers are separate from the final backing charge.
    ///
    /// # Errors
    /// Rejects zero-sized elements, invalid stride/alignment, size overflow and
    /// empty mapped payloads. Propagates callback, budget and storage failures;
    /// failure discards the partial backing and releases its reservation.
    pub fn build_rows<E: From<io::Error>>(
        rows: usize,
        stride: usize,
        alignment: usize,
        options: &RegionOptions,
        mut fill: impl FnMut(usize, &mut [T]) -> Result<(), E>,
    ) -> Result<Self, E> {
        let row_bytes = stride
            .checked_mul(size_of::<T>())
            .filter(|n| *n > 0)
            .ok_or_else(|| invalid("invalid region row size"))?;
        let bytes = rows
            .checked_mul(row_bytes)
            .filter(|n| *n <= isize::MAX as usize)
            .ok_or_else(|| invalid("region length overflow"))?;
        let alignment = alignment.max(align_of::<T>());
        let lease = options
            .accounting
            .as_ref()
            .map(|a| a.reserve(bytes, options.placement))
            .transpose()?;
        let backing = match options.placement {
            Placement::Resident => {
                let mut owned = Allocation::new(bytes, alignment)?;
                fill(0, bytemuck::cast_slice_mut(owned.bytes_mut(bytes)))?;
                Backing::Owned(owned)
            }
            Placement::Mapped => {
                #[cfg(target_arch = "wasm32")]
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "file mappings require a native filesystem",
                )
                .into());
                #[cfg(not(target_arch = "wasm32"))]
                {
                    use std::io::Write;
                    if bytes == 0 {
                        return Err(invalid("cannot map an empty payload").into());
                    }
                    let mut file = match &options.directory {
                        Some(dir) => tempfile::tempfile_in(dir)?,
                        None => tempfile::tempfile()?,
                    };
                    let chunk_rows = (crate::buffered_io::FILE_WRITE_BYTES / row_bytes)
                        .max(1)
                        .min(rows);
                    let mut block = Allocation::new(chunk_rows * row_bytes, alignment)?;
                    let mut writer = std::io::BufWriter::with_capacity(
                        crate::buffered_io::FILE_WRITE_BYTES,
                        &mut file,
                    );
                    let mut reclaimed = 0usize;
                    for start in (0..rows).step_by(chunk_rows) {
                        let count = chunk_rows.min(rows - start);
                        let buf = block.bytes_mut(count * row_bytes);
                        buf.fill(0);
                        fill(start, bytemuck::cast_slice_mut(buf))?;
                        writer.write_all(buf)?;
                        let end = (start + count) * row_bytes;
                        if options.reclaim_after_write
                            && (end - reclaimed >= crate::file_cache::RECLAIM_WINDOW_BYTES
                                || end == bytes)
                        {
                            writer.flush()?;
                            crate::file_cache::sync_and_reclaim_file(
                                writer.get_ref(),
                                reclaimed as u64..end as u64,
                            )?;
                            reclaimed = end;
                        }
                    }
                    writer.flush()?;
                    drop(writer);
                    // SAFETY: this anonymous file is created here and no writable
                    // handle or external name escapes. Mapping owns its lifetime.
                    let map = unsafe { memmap2::Mmap::map(&file)? };
                    if !map.as_ptr().addr().is_multiple_of(alignment) {
                        return Err(invalid("mapping alignment unsupported").into());
                    }
                    #[cfg(unix)]
                    if let Err(e) = map.advise(memmap2::Advice::Random) {
                        log::debug!("mapping advice unavailable: {e}");
                    }
                    Backing::Mapped { map, file }
                }
            }
        };
        let region = ReadRegion {
            backing: Arc::new(AccountedBacking {
                len: bytes,
                backing,
                lease,
            }),
            offset: 0,
            len: bytes,
        };
        Ok(region.array()?)
    }
}

/// Classifies region layout or bounds violations as invalid-input I/O errors.
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
