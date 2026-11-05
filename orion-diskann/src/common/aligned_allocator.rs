/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::alloc::Layout;
use std::ops::{Deref, DerefMut, Range};
use std::ptr::NonNull;

use super::{ANNError, ANNResult};

/// Owned, aligned storage for plain data. The allocation is freed with exactly
/// the layout used to create it, including any Linux huge-page alignment.
///
/// Zero initialization is only available for POD elements:
/// ```compile_fail
/// use orion_ann_diskann::common::AlignedBoxWithSlice;
/// let _ = AlignedBoxWithSlice::<std::num::NonZeroU32>::new(1, 64);
/// ```
pub struct AlignedBoxWithSlice<T> {
    layout: Layout,
    ptr: NonNull<T>,
    len: usize,
}

// SAFETY: this allocation has unique ownership. Shared access exposes only &T,
// and mutation requires &mut self, just as for Vec<T>.
unsafe impl<T: Send> Send for AlignedBoxWithSlice<T> {}
unsafe impl<T: Sync> Sync for AlignedBoxWithSlice<T> {}

impl<T> AlignedBoxWithSlice<T> {
    /// Allocate zeroed POD elements. A smaller requested alignment is raised to
    /// the element alignment. Empty and zero-sized slices do not allocate.
    ///
    /// # Errors
    /// Rejects non-power-of-two/zero alignment, layout overflow and allocation failure.
    pub fn new(capacity: usize, alignment: usize) -> ANNResult<Self>
    where
        T: bytemuck::Pod,
    {
        // Validate the caller's request before raising it to T's alignment.
        Layout::from_size_align(0, alignment).map_err(ANNError::log_mem_alloc_layout_error)?;
        let alignment = alignment.max(std::mem::align_of::<T>());
        let allocsize = capacity
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ANNError::log_index_error("capacity overflow".to_string()))?;
        let layout = Layout::from_size_align(allocsize, alignment)
            .map_err(ANNError::log_mem_alloc_layout_error)?;

        #[cfg(target_os = "linux")]
        let layout = if allocsize >= orion_ann_platform::memory::HUGE_PAGE_ALIGNMENT {
            // Keep the requested length, but make large allocations eligible
            // for THP from their first page. Drop uses this same Layout.
            Layout::from_size_align(
                allocsize,
                alignment.max(orion_ann_platform::memory::HUGE_PAGE_ALIGNMENT),
            )
            .map_err(ANNError::log_mem_alloc_layout_error)?
        } else {
            layout
        };

        if layout.size() == 0 {
            // A slice needs an aligned, non-null pointer even when it has no bytes.
            let ptr = NonNull::new(std::ptr::without_provenance_mut::<T>(layout.align())).unwrap();
            return Ok(Self {
                layout,
                ptr,
                len: capacity,
            });
        }

        // SAFETY: the nonzero Layout covers every element and its alignment.
        // Pod permits zeroed values; allocation failure is checked before access.
        let ptr = unsafe {
            #[cfg(target_os = "linux")]
            let use_huge_pages = allocsize >= orion_ann_platform::memory::HUGE_PAGE_ALIGNMENT;
            #[cfg(not(target_os = "linux"))]
            let use_huge_pages = false;

            // For large Linux buffers, advise before zeroing. alloc_zeroed
            // could otherwise fault in ordinary pages before the hint arrives.
            let mem = if use_huge_pages {
                std::alloc::alloc(layout)
            } else {
                std::alloc::alloc_zeroed(layout)
            };
            if mem.is_null() {
                return Err(ANNError::log_index_error(format!(
                    "allocation failed: {allocsize} bytes"
                )));
            }
            if use_huge_pages {
                #[cfg(target_os = "linux")]
                if let Err(error) = orion_ann_platform::memory::advise_huge_pages(mem, allocsize) {
                    log::warn!(
                        "THP advice failed for {allocsize} bytes: {error}; using ordinary pages"
                    );
                }
                std::ptr::write_bytes(mem, 0, allocsize);
            }
            NonNull::new_unchecked(mem.cast::<T>())
        };

        Ok(Self {
            layout,
            ptr,
            len: capacity,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: all len elements are initialized and live for this borrow.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: &mut self excludes all other access to the owned allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Copy into the start of this allocation, retaining the remaining elements.
    pub fn memcpy(&mut self, src: &[T]) -> ANNResult<()>
    where
        T: Copy,
    {
        if src.len() > self.len {
            return Err(ANNError::log_index_error(format!(
                "source slice is too large (src:{}, dst:{})",
                src.len(),
                self.len
            )));
        }
        // Safe borrows already rule out overlapping source and destination.
        self.as_mut_slice()[..src.len()].copy_from_slice(src);
        Ok(())
    }

    pub fn split_into_nonoverlapping_mut_slices(
        &mut self,
        range: Range<usize>,
        slice_len: usize,
    ) -> ANNResult<Vec<&mut [T]>> {
        if slice_len == 0
            || range.start > range.end
            || range.end > self.len
            || range.len() % slice_len != 0
        {
            return Err(ANNError::log_index_error(format!(
                "invalid slice range {range:?}, chunk length {slice_len}, allocation length {}",
                self.len
            )));
        }
        Ok(self.as_mut_slice()[range]
            .chunks_exact_mut(slice_len)
            .collect())
    }
}

impl<T> Drop for AlignedBoxWithSlice<T> {
    fn drop(&mut self) {
        // SAFETY: the elements are initialized and uniquely owned; nonempty
        // allocations are freed exactly once using their original Layout.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(
                self.ptr.as_ptr(),
                self.len,
            ));
            if self.layout.size() != 0 {
                std::alloc::dealloc(self.ptr.as_ptr().cast::<u8>(), self.layout);
            }
        }
    }
}

impl<T> Deref for AlignedBoxWithSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedBoxWithSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

// Log allocation metadata, never the potentially multi-terabyte payload.
impl<T> std::fmt::Debug for AlignedBoxWithSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBoxWithSlice")
            .field("element_type", &std::any::type_name::<T>())
            .field("len", &self.len)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_zero_sized_slices_keep_requested_alignment() {
        let mut empty = AlignedBoxWithSlice::<u64>::new(0, 128).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.as_ptr().addr() % 128, 0);
        empty.memcpy(&[]).unwrap();
        assert!(
            empty
                .split_into_nonoverlapping_mut_slices(0..0, 1)
                .unwrap()
                .is_empty()
        );

        let mut units = AlignedBoxWithSlice::<()>::new(7, 64).unwrap();
        assert_eq!(units.len(), 7);
        assert_eq!(units.as_ptr().addr() % 64, 0);
        units.memcpy(&[(); 7]).unwrap();
        assert_eq!(
            units
                .split_into_nonoverlapping_mut_slices(1..7, 2)
                .unwrap()
                .len(),
            3
        );
    }

    #[test]
    fn element_alignment_is_a_lower_bound() {
        let mut values = AlignedBoxWithSlice::<u64>::new(3, 1).unwrap();
        assert_eq!(values.as_ptr().addr() % std::mem::align_of::<u64>(), 0);
        values.memcpy(&[42, 17]).unwrap();
        assert_eq!(values.as_slice(), &[42, 17, 0]);
    }

    #[test]
    fn allocation_can_be_moved_and_shared_between_threads() {
        let mut values = AlignedBoxWithSlice::<u32>::new(4, 64).unwrap();
        values.memcpy(&[1, 2, 3, 4]).unwrap();
        let values = std::thread::spawn(move || values).join().unwrap();
        std::thread::scope(|scope| {
            let sum = scope.spawn(|| values.iter().sum::<u32>());
            assert_eq!(sum.join().unwrap(), 10);
        });
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn large_allocation_preserves_length_zeroing_and_alignment() {
        // An odd size exercises the unadvised partial final page as well.
        let size = orion_ann_platform::memory::HUGE_PAGE_ALIGNMENT + 17;
        let mut data = AlignedBoxWithSlice::<u8>::new(size, 64).unwrap();
        assert_eq!(data.len(), size);
        assert_eq!(
            data.as_ptr().addr() % orion_ann_platform::memory::HUGE_PAGE_ALIGNMENT,
            0
        );
        assert!(data.iter().all(|&value| value == 0));
        data[0] = 7;
        data[size - 1] = 9;
        assert_eq!((data[0], data[size - 1]), (7, 9));
        drop(data); // Must free with the upgraded allocation layout.
    }

    #[test]
    fn create_alignedvec_works_32() {
        let size = 1000;
        let data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        assert_eq!(data.len(), size);
        let ptr = data.as_ptr() as usize;
        assert_eq!(ptr % 32, 0);
        (0..size).for_each(|i| {
            assert_eq!(data[i], f32::default());
        });
    }

    #[test]
    fn split_into_nonoverlapping_mut_slices_test() {
        let size = 10;
        let slice_len = 2;
        let mut data = AlignedBoxWithSlice::<f32>::new(size, 32).unwrap();
        let slices = data
            .split_into_nonoverlapping_mut_slices(2..8, slice_len)
            .unwrap();
        assert_eq!(slices.len(), 3);
        for (i, slice) in slices.into_iter().enumerate() {
            assert_eq!(slice.len(), slice_len);
            slice[0] = i as f32 + 1.0;
            slice[1] = i as f32 + 1.0;
        }
        let expected_arr = [0.0f32, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 0.0, 0.0];
        assert_eq!(data.as_ref(), &expected_arr);
    }
}
