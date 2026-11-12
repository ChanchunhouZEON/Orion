/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Immutable storage ownership, independent of object formats and search semantics.
//! A view owns its backing through Arc. Construction is exclusive; publishing a
//! view removes all writable access. File mappings only cover private temporary
//! files: callers cannot invalidate a slice by overwriting an external cache.
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
    #[default]
    Resident,
    Mapped,
}

/// Private backing-file directory. None uses the OS temporary directory.
/// Prefer a disk-backed directory when using mappings to reduce memory pressure.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RegionOptions {
    pub placement: Placement,
    pub directory: Option<PathBuf>,
}

struct Allocation {
    ptr: NonNull<u8>,
    layout: Layout,
}
impl Allocation {
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
    fn bytes_mut(&mut self, len: usize) -> &mut [u8] {
        // SAFETY: initialized allocation, exclusive borrow, bounded by constructor.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), len) }
    }
}
impl Drop for Allocation {
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
}
enum MutableBacking {
    Owned(Allocation),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::MmapMut),
}
// SAFETY: byte access follows ordinary shared/exclusive borrowing. Backing is
// uniquely owned and no other handle can mutate or truncate its private inode.
unsafe impl Send for MutableRegion {}
unsafe impl Sync for MutableRegion {}
impl std::fmt::Debug for MutableRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MutableRegion")
            .field("bytes", &self.len)
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}
impl MutableRegion {
    /// Allocate zero-initialized bytes, either aligned RAM or a private mapping.
    /// # Errors
    /// Rejects invalid layouts and reports allocation, filesystem or mapping failures.
    pub fn new(bytes: usize, alignment: usize, options: &RegionOptions) -> io::Result<Self> {
        Layout::from_size_align(bytes, alignment).map_err(|_| invalid("invalid region layout"))?;
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
                MutableBacking::Mapped(mapping)
            }
        };
        let ptr = match &backing {
            MutableBacking::Owned(a) => a.ptr,
            #[cfg(not(target_arch = "wasm32"))]
            MutableBacking::Mapped(m) => NonNull::new(m.as_ptr().cast_mut())
                .ok_or_else(|| invalid("nonempty mapping has a null address"))?,
        };
        Ok(Self {
            ptr,
            len: bytes,
            backing,
        })
    }
    #[inline(always)]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: initialized, owned bytes; cached length is validated on construction.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    #[inline(always)]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: unique borrow excludes every other reference to these bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
    pub fn placement(&self) -> Placement {
        match self.backing {
            MutableBacking::Owned(_) => Placement::Resident,
            #[cfg(not(target_arch = "wasm32"))]
            MutableBacking::Mapped(_) => Placement::Mapped,
        }
    }
    /// Finish construction without copying, dropping all writable access.
    pub fn freeze(self) -> io::Result<ReadRegion> {
        let backing = match self.backing {
            MutableBacking::Owned(a) => Backing::Owned(a),
            #[cfg(not(target_arch = "wasm32"))]
            MutableBacking::Mapped(m) => Backing::Mapped(m.make_read_only()?),
        };
        Ok(ReadRegion {
            backing: Arc::new(backing),
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

enum Backing {
    Owned(Allocation),
    #[cfg(not(target_arch = "wasm32"))]
    Mapped(memmap2::Mmap),
}
impl Backing {
    fn ptr(&self) -> *const u8 {
        match self {
            Self::Owned(a) => a.ptr.as_ptr(),
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped(m) => m.as_ptr(),
        }
    }
    fn placement(&self) -> Placement {
        match self {
            Self::Owned(_) => Placement::Resident,
            #[cfg(not(target_arch = "wasm32"))]
            Self::Mapped(_) => Placement::Mapped,
        }
    }
}

/// An immutable byte range owning the backing, including after its parent drops.
#[derive(Clone)]
pub struct ReadRegion {
    backing: Arc<Backing>,
    offset: usize,
    len: usize,
}
impl std::fmt::Debug for ReadRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReadRegion")
            .field("bytes", &self.len)
            .field("placement", &self.placement())
            .finish_non_exhaustive()
    }
}
impl ReadRegion {
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: constructors/range validate length, backing lives with the view.
        unsafe { std::slice::from_raw_parts(self.backing.ptr().add(self.offset), self.len) }
    }
    pub fn placement(&self) -> Placement {
        self.backing.placement()
    }
    /// Borrow a section without copying; malformed ranges return InvalidInput.
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
    /// Interpret initialized POD bytes, checking element size and alignment.
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.region.fmt(f)
    }
}
impl<T: Pod> std::ops::Deref for ArrayView<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T: Pod> ArrayView<T> {
    /// Copy a borrowed payload into independently owned storage.
    pub fn copy_from(values: &[T], options: &RegionOptions) -> io::Result<Self> {
        Self::build_rows(values.len(), 1, 64, options, |start, out| {
            out.copy_from_slice(&values[start..start + out.len()]);
            Ok(())
        })
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the checked view owns immutable, aligned POD storage. Its
        // cached pointer and length cannot change independently of that owner.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn region(&self) -> &ReadRegion {
        &self.region
    }
    pub fn placement(&self) -> Placement {
        self.region.placement()
    }
    /// Initialize rows before publication. Mapped placement uses a bounded block
    /// and writes to a private file; no full-sized heap or mutable mmap is made.
    /// `fill` receives the first row index and zero-initialized complete rows.
    /// Errors discard the partial backing. The callback must initialize logical
    /// values; unused row padding remains zero. Zero-sized element types rejected.
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
                    for start in (0..rows).step_by(chunk_rows) {
                        let count = chunk_rows.min(rows - start);
                        let buf = block.bytes_mut(count * row_bytes);
                        buf.fill(0);
                        fill(start, bytemuck::cast_slice_mut(buf))?;
                        writer.write_all(buf)?;
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
                    Backing::Mapped(map)
                }
            }
        };
        let region = ReadRegion {
            backing: Arc::new(backing),
            offset: 0,
            len: bytes,
        };
        Ok(region.array()?)
    }
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
