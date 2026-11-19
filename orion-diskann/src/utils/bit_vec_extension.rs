/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::cmp::Ordering;

use bit_vec::BitVec;

/// Resizes legacy visited-bit storage while preserving existing bits.
pub trait BitVecExtension {
    /// Changes bit-vector length, filling newly added bits with the requested value.
    fn resize(&mut self, new_len: usize, value: bool);
}

impl BitVecExtension for BitVec {
    /// Truncates the bit vector or fills newly appended bits with the supplied value.
    fn resize(&mut self, new_len: usize, value: bool) {
        let old_len = self.len();
        match new_len.cmp(&old_len) {
            Ordering::Less => self.truncate(new_len),
            Ordering::Greater => self.grow(new_len - old_len, value),
            Ordering::Equal => {}
        }
    }
}
