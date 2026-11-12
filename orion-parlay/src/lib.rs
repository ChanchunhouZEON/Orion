/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! In-memory ParlayANN construction. Direct builds borrow input rows and the
//! final output slab synchronously. The owning reference builder borrows input
//! until dropped; its partitions borrow a bounded batch until the next operation.
//! C++ exceptions are caught after parallel jobs join, never across this ABI.
use std::{
    ffi::{c_void, CStr, CString},
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
};

/// Stable bridge failure category, mapped by the runtime into Orion's errors.
#[derive(Debug)]
pub struct Error {
    pub code: u32,
    pub message: String,
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;
fn invalid(message: &str) -> Error {
    Error {
        code: 1,
        message: message.into(),
    }
}

/// Construction recipe. Versioned in the runtime cache provenance.
#[derive(Clone, Copy, Debug)]
pub struct BuildOptions {
    pub degree: u32,
    pub build_l: u32,
    pub max_extra: u32,
    pub threads: u32,
    pub local_pct: u32,
    pub passes: u32,
    pub alpha: f64,
    pub metric: Metric,
}
#[derive(Clone, Copy, Debug)]
pub enum Metric {
    L2,
    InnerProduct,
    /// Cosine over native u8 with wide, exact squared norms.
    Cosine,
}
impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            degree: 64,
            build_l: 100,
            max_extra: 16,
            threads: 8,
            local_pct: 60,
            passes: 2,
            alpha: 1.2,
            metric: Metric::L2,
        }
    }
}
#[repr(C)]
struct Options {
    count: usize,
    dimension: usize,
    stride: usize,
    degree: u32,
    build_l: u32,
    max_extra: u32,
    threads: u32,
    local_pct: u32,
    passes: u32,
    metric: u32,
    alpha: f64,
}
impl Options {
    fn new(count: usize, dimension: usize, stride: usize, o: BuildOptions) -> Self {
        Self {
            count,
            dimension,
            stride,
            degree: o.degree,
            build_l: o.build_l,
            max_extra: o.max_extra,
            threads: o.threads,
            local_pct: o.local_pct,
            passes: o.passes,
            metric: match o.metric {
                Metric::L2 => 0,
                Metric::InnerProduct => 1,
                Metric::Cosine => 2,
            },
            alpha: o.alpha,
        }
    }
}
#[repr(C)]
struct Failure {
    code: u32,
    message: [std::ffi::c_char; 1024],
}
impl Failure {
    fn new() -> Self {
        Self {
            code: 0,
            message: [0; 1024],
        }
    }
    fn check(self, status: i32) -> Result<()> {
        if status == 0 {
            return Ok(());
        }
        // The C++ boundary always initializes and NUL-terminates this array.
        Err(Error {
            code: self.code,
            message: unsafe { CStr::from_ptr(self.message.as_ptr()) }
                .to_string_lossy()
                .into_owned(),
        })
    }
}
#[repr(C)]
struct RawPartition {
    local: *const u32,
    remote: *const u32,
    extra: *const u32,
    local_len: usize,
    remote_len: usize,
    extra_len: usize,
}
unsafe extern "C" {
    fn orion_pa_build(
        data: *const c_void,
        bytes: bool,
        options: *const Options,
        result: *mut *mut c_void,
        error: *mut Failure,
    ) -> i32;
    fn orion_pa_build_into(
        data: *const c_void,
        bytes: bool,
        options: *const Options,
        slab: *mut u32,
        words: usize,
        stride: usize,
        error: *mut Failure,
    ) -> i32;
    fn orion_pa_next(
        handle: *mut c_void,
        part: *mut RawPartition,
        present: *mut bool,
        error: *mut Failure,
    ) -> i32;
    fn orion_pa_save(
        handle: *mut c_void,
        path: *const std::ffi::c_char,
        reference: bool,
        error: *mut Failure,
    ) -> i32;
    fn orion_pa_free(handle: *mut c_void);
    #[cfg(test)]
    fn orion_pa_test_exception(error: *mut Failure) -> i32;
}
/// Distance-sorted navigation and residual candidate partitions for one node.
pub struct Partition<'a> {
    pub local: &'a [u32],
    pub remote: &'a [u32],
    pub extra: &'a [u32],
}
/// Owns the C++ graph/extras, borrowing the immutable base allocation.
/// This handle cannot be shared across threads. Separate builds use independent
/// schedulers. All operations are synchronous and obey the configured threads.
pub struct Builder<'a> {
    handle: NonNull<c_void>,
    _base: PhantomData<&'a [u8]>,
}
impl Drop for Builder<'_> {
    fn drop(&mut self) {
        unsafe { orion_pa_free(self.handle.as_ptr()) };
    }
}
impl<'a> Builder<'a> {
    /// Build from f32 rows. Padding is skipped; stride is measured in elements.
    ///
    /// # Errors
    /// Invalid shapes/coordinates/options and C++ failures return `Error`.
    pub fn f32(
        base: &'a [f32],
        count: usize,
        dimension: usize,
        stride: usize,
        options: BuildOptions,
    ) -> Result<Self> {
        Self::shape(base.len(), count, dimension, stride)?;
        if base
            .chunks(stride)
            .take(count)
            .any(|row| row[..dimension].iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("base contains a non-finite coordinate"));
        }
        Self::build(
            base.as_ptr().cast(),
            false,
            count,
            dimension,
            stride,
            options,
        )
    }
    /// Build native-u8 L2, IP or cosine without conversion or quantization.
    ///
    /// # Errors
    /// Rejects invalid shapes, unsupported metrics/distance accumulator sizes,
    /// invalid controls and recoverable backend allocation/build failures.
    pub fn u8(
        base: &'a [u8],
        count: usize,
        dimension: usize,
        stride: usize,
        options: BuildOptions,
    ) -> Result<Self> {
        Self::shape(base.len(), count, dimension, stride)?;
        Self::build(
            base.as_ptr().cast(),
            true,
            count,
            dimension,
            stride,
            options,
        )
    }
    /// Build and partition directly into caller-owned Orion rows.
    ///
    /// Rows use four header words followed by neighbors/extras; slab stride is
    /// measured in u32 words. All worker jobs finish before returning. On error
    /// the slab may be partially written and must not be published as a graph.
    ///
    /// # Errors
    /// Rejects invalid input/layout/options and propagates joined C++ failures.
    pub fn f32_into(
        base: &[f32],
        count: usize,
        dimension: usize,
        stride: usize,
        options: BuildOptions,
        slab: &mut [u32],
        slab_stride: usize,
    ) -> Result<()> {
        Self::shape(base.len(), count, dimension, stride)?;
        if base
            .chunks(stride)
            .take(count)
            .any(|r| r[..dimension].iter().any(|x| !x.is_finite()))
        {
            return Err(invalid("base contains a non-finite coordinate"));
        }
        Self::into_slab(
            base.as_ptr().cast(),
            false,
            Options::new(count, dimension, stride, options),
            slab,
            slab_stride,
        )
    }
    /// Native-u8 variant of [`Self::f32_into`], without a converted base buffer.
    ///
    /// # Errors
    /// Invalid shapes/metrics/layouts and backend failures return `Error`.
    /// A failed call leaves the output unpublished and possibly partially written.
    pub fn u8_into(
        base: &[u8],
        count: usize,
        dimension: usize,
        stride: usize,
        options: BuildOptions,
        slab: &mut [u32],
        slab_stride: usize,
    ) -> Result<()> {
        Self::shape(base.len(), count, dimension, stride)?;
        Self::into_slab(
            base.as_ptr().cast(),
            true,
            Options::new(count, dimension, stride, options),
            slab,
            slab_stride,
        )
    }
    fn into_slab(
        data: *const c_void,
        bytes: bool,
        options: Options,
        slab: &mut [u32],
        stride: usize,
    ) -> Result<()> {
        let mut error = Failure::new();
        // Input and output come from disjoint shared/exclusive Rust borrows.
        // C++ validates row capacity, retains neither pointer, and joins workers.
        let status = unsafe {
            orion_pa_build_into(
                data,
                bytes,
                &options,
                slab.as_mut_ptr(),
                slab.len(),
                stride,
                &mut error,
            )
        };
        error.check(status)
    }
    fn shape(len: usize, count: usize, dimension: usize, stride: usize) -> Result<()> {
        if count == 0
            || dimension == 0
            || stride < dimension
            || count.checked_mul(stride).is_none_or(|n| n > len)
        {
            return Err(invalid("base length/count/dimension/stride mismatch"));
        }
        Ok(())
    }
    fn build(
        data: *const c_void,
        bytes: bool,
        count: usize,
        dimension: usize,
        stride: usize,
        o: BuildOptions,
    ) -> Result<Self> {
        let options = Options::new(count, dimension, stride, o);
        let mut handle = std::ptr::null_mut();
        let mut error = Failure::new();
        // Both public constructors tie this handle to the borrowed input lifetime.
        let status = unsafe { orion_pa_build(data, bytes, &options, &mut handle, &mut error) };
        error.check(status)?;
        Ok(Self {
            handle: NonNull::new(handle).ok_or_else(|| invalid("builder returned null"))?,
            _base: PhantomData,
        })
    }
    /// Return the next node. Slices remain valid until the next mutable operation.
    ///
    /// # Errors
    /// Propagates partition computation failures. Drop the builder after error.
    pub fn next_partition(&mut self) -> Result<Option<Partition<'_>>> {
        let mut part = RawPartition {
            local: std::ptr::null(),
            remote: std::ptr::null(),
            extra: std::ptr::null(),
            local_len: 0,
            remote_len: 0,
            extra_len: 0,
        };
        let mut present = false;
        let mut error = Failure::new();
        let status =
            unsafe { orion_pa_next(self.handle.as_ptr(), &mut part, &mut present, &mut error) };
        error.check(status)?;
        if !present {
            return Ok(None);
        }
        // Empty C++ vectors may have null data(); Rust empty slices may not.
        unsafe fn slice<'b>(p: *const u32, n: usize) -> &'b [u32] {
            if n == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(p, n) }
            }
        }
        Ok(Some(Partition {
            local: unsafe { slice(part.local, part.local_len) },
            remote: unsafe { slice(part.remote, part.remote_len) },
            extra: unsafe { slice(part.extra, part.extra_len) },
        }))
    }
    /// Export STAG v3 using exactly the same partition computation as memory output.
    ///
    /// # Errors
    /// Invalid path encoding, write/publish errors or partition failures.
    pub fn save_staged(&mut self, path: &Path) -> Result<()> {
        self.save(path, false)
    }
    /// Frozen pre-bridge STAG writer for equivalence tests only.
    #[cfg(feature = "reference-tests")]
    pub fn save_reference(&mut self, path: &Path) -> Result<()> {
        self.save(path, true)
    }
    fn save(&mut self, path: &Path, reference: bool) -> Result<()> {
        let path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| invalid("path contains NUL"))?;
        let mut error = Failure::new();
        let status =
            unsafe { orion_pa_save(self.handle.as_ptr(), path.as_ptr(), reference, &mut error) };
        error.check(status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concurrent_borrowed_builds_are_deterministic_and_release_cleanly() {
        let n = 96;
        let dim = 13;
        let stride = 32;
        let bytes: Vec<u8> = (0..n * stride)
            .map(|i| ((i * 73 + i / stride) % 251) as u8)
            .collect();
        let floats: Vec<f32> = bytes.iter().map(|&x| f32::from(x)).collect();
        let opts = BuildOptions {
            degree: 8,
            build_l: 16,
            max_extra: 4,
            threads: 2,
            ..BuildOptions::default()
        };
        fn collect(mut b: Builder<'_>, n: usize) -> Vec<Vec<u32>> {
            let mut rows = Vec::new();
            while let Some(p) = b.next_partition().unwrap() {
                let row: Vec<_> = p
                    .local
                    .iter()
                    .chain(p.remote)
                    .chain(p.extra)
                    .copied()
                    .collect();
                assert!(row.iter().all(|&id| (id as usize) < n));
                rows.push(row);
            }
            assert_eq!(rows.len(), n);
            rows
        }
        std::thread::scope(|scope| {
            let a =
                scope.spawn(|| collect(Builder::f32(&floats, n, dim, stride, opts).unwrap(), n));
            let b = scope.spawn(|| collect(Builder::u8(&bytes, n, dim, stride, opts).unwrap(), n));
            assert_eq!(a.join().unwrap(), b.join().unwrap());
        });
        let expected = collect(Builder::u8(&bytes, n, dim, stride, opts).unwrap(), n);
        let slab_stride = 32;
        let direct = |native: bool| {
            let mut storage = vec![0xdeadbeefu32; n * slab_stride + 2];
            let slab = &mut storage[1..1 + n * slab_stride];
            if native {
                Builder::u8_into(&bytes, n, dim, stride, opts, slab, slab_stride).unwrap();
            } else {
                Builder::f32_into(&floats, n, dim, stride, opts, slab, slab_stride).unwrap();
            }
            for (row, ids) in slab.chunks_exact(slab_stride).zip(&expected) {
                assert_eq!(&row[4..4 + row[0] as usize + row[1] as usize], ids);
                assert!(row[2] <= row[0]);
                assert_eq!(row[3], 0);
                assert!(row[4 + row[0] as usize + row[1] as usize..]
                    .iter()
                    .all(|&x| x == 0));
            }
            assert_eq!(storage[0], 0xdeadbeef);
            assert_eq!(*storage.last().unwrap(), 0xdeadbeef);
        };
        // Concurrent slab builds must keep separate schedulers and borrows;
        // padding and out-of-slice guards must survive the temporary layout.
        std::thread::scope(|scope| {
            scope.spawn(|| direct(true));
            scope.spawn(|| direct(false));
        });
    }

    #[test]
    fn worker_exception_is_returned_after_join() {
        let mut error = Failure::new();
        let status = unsafe { orion_pa_test_exception(&mut error) };
        assert!(error
            .check(status)
            .unwrap_err()
            .message
            .contains("worker failure probe"));
    }
    #[test]
    fn reject_shapes_and_cpp_options() {
        let data = vec![0.; 16];
        assert!(Builder::f32(&data, 8, 3, 4, BuildOptions::default()).is_err());
        let opts = BuildOptions {
            threads: 0,
            ..BuildOptions::default()
        };
        assert_eq!(Builder::f32(&data, 4, 4, 4, opts).err().unwrap().code, 1);
        assert!(Builder::f32(&[f32::NAN; 16], 4, 4, 4, BuildOptions::default()).is_err());
    }
}
