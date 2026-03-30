/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */
use crate::common::{ANNError, ANNResult};

pub const DISK_IO_ALIGNMENT: usize = 512;

/// Aligned read struct for disk IO, it takes the ownership of the AlignedBoxedSlice and returns the AlignedBoxWithSlice data immutably.
pub struct AlignedRead<'a, T>
where
    T: Copy,
{
    /// where to read from
    /// offset needs to be aligned with DISK_IO_ALIGNMENT
    pub offset: u64,

    /// where to read into
    /// aligned_buf and its len need to be aligned with DISK_IO_ALIGNMENT
    pub aligned_buf: &'a mut [T],
}

impl<'a, T: Copy> AlignedRead<'a, T> {
    pub fn new(offset: u64, aligned_buf: &'a mut [T]) -> ANNResult<Self> {
        Self::assert_is_aligned(offset as usize)?;
        Self::assert_is_aligned(std::mem::size_of_val(aligned_buf))?;

        Ok(Self {
            offset,
            aligned_buf,
        })
    }

    fn assert_is_aligned(val: usize) -> ANNResult<()> {
        match val % DISK_IO_ALIGNMENT {
            0 => Ok(()),
            _ => Err(ANNError::log_disk_io_request_alignment_error(format!(
                "The offset or length of AlignedRead request is not {} bytes aligned",
                DISK_IO_ALIGNMENT
            ))),
        }
    }

    pub fn aligned_buf(&self) -> &[T] {
        self.aligned_buf
    }
}
