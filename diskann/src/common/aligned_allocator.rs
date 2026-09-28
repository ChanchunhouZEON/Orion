/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::alloc::Layout;
use std::ops::{Deref, DerefMut, Range};
use std::ptr::copy_nonoverlapping;

use super::{ANNError, ANNResult};

#[derive(Debug)]
pub struct AlignedBoxWithSlice<T> {
    layout: Layout,
    val: Box<[T]>,
}

impl<T> AlignedBoxWithSlice<T> {
    pub fn new(capacity: usize, alignment: usize) -> ANNResult<Self> {
        let allocsize = capacity
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| ANNError::log_index_error("capacity overflow".to_string()))?;
        let layout = Layout::from_size_align(allocsize, alignment)
            .map_err(ANNError::log_mem_alloc_layout_error)?;

        let val = unsafe {
            let mem = std::alloc::alloc_zeroed(layout);
            if mem.is_null() {
                return Err(ANNError::log_index_error(format!(
                    "allocation failed: {allocsize} bytes"
                )));
            }
            let ptr = mem as *mut T;
            let slice = std::slice::from_raw_parts_mut(ptr, capacity);
            std::boxed::Box::from_raw(slice)
        };

        Ok(Self { layout, val })
    }

    pub fn as_slice(&self) -> &[T] {
        &self.val
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.val
    }

    pub fn memcpy(&mut self, src: &[T]) -> ANNResult<()> {
        if src.len() > self.val.len() {
            return Err(ANNError::log_index_error(format!(
                "source slice is too large (src:{}, dst:{})",
                src.len(),
                self.val.len()
            )));
        }

        let src_ptr = src.as_ptr();
        let src_end = unsafe { src_ptr.add(src.len()) };
        let dst_ptr = self.val.as_mut_ptr();
        let dst_end = unsafe { dst_ptr.add(self.val.len()) };

        if src_ptr < dst_end && src_end > dst_ptr {
            return Err(ANNError::log_index_error(
                "Source and destination overlap".to_string(),
            ));
        }

        unsafe {
            copy_nonoverlapping(src.as_ptr(), self.val.as_mut_ptr(), src.len());
        }

        Ok(())
    }

    pub fn split_into_nonoverlapping_mut_slices(
        &mut self,
        range: Range<usize>,
        slice_len: usize,
    ) -> ANNResult<Vec<&mut [T]>> {
        if range.len() % slice_len != 0 || range.end > self.len() {
            return Err(ANNError::log_index_error(format!(
                "Cannot split range ({:?}) of AlignedBoxWithSlice (len: {}) into nonoverlapping mutable slices with length {}",
                range,
                self.len(),
                slice_len,
            )));
        }

        let mut slices = Vec::with_capacity(range.len() / slice_len);
        let mut remaining_slice = &mut self.val[range];

        while remaining_slice.len() >= slice_len {
            let (left, right) = remaining_slice.split_at_mut(slice_len);
            slices.push(left);
            remaining_slice = right;
        }

        Ok(slices)
    }
}

impl<T> Drop for AlignedBoxWithSlice<T> {
    fn drop(&mut self) {
        let val = std::mem::take(&mut self.val);
        let mut val2 = std::mem::ManuallyDrop::new(val);
        let ptr = val2.as_mut_ptr();

        unsafe { std::alloc::dealloc(ptr as *mut u8, self.layout) }
    }
}

impl<T> Deref for AlignedBoxWithSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<T> DerefMut for AlignedBoxWithSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
