/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use std::cmp::Ordering;

use bit_vec::BitVec;

pub trait BitVecExtension {
    fn resize(&mut self, new_len: usize, value: bool);
}

impl BitVecExtension for BitVec {
    fn resize(&mut self, new_len: usize, value: bool) {
        let old_len = self.len();
        match new_len.cmp(&old_len) {
            Ordering::Less => self.truncate(new_len),
            Ordering::Greater => self.grow(new_len - old_len, value),
            Ordering::Equal => {}
        }
    }
}
