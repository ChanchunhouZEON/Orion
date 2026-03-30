/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use bytemuck::{Pod, Zeroable};
use half::f16;
use std::convert::AsRef;
use std::fmt;

/// Half-precision floating point wrapper around f16.
/// Memory layout is identical to f16 (2 bytes, 2-byte aligned).
pub struct Half(f16);

unsafe impl Pod for Half {}
unsafe impl Zeroable for Half {}

impl From<Half> for f32 {
    fn from(val: Half) -> Self {
        val.0.to_f32()
    }
}

impl AsRef<f16> for Half {
    fn as_ref(&self) -> &f16 {
        &self.0
    }
}

impl Half {
    pub fn from_f32(value: f32) -> Self {
        Self(f16::from_f32(value))
    }

    pub fn to_f32(&self) -> f32 {
        self.0.to_f32()
    }
}

impl Default for Half {
    fn default() -> Self {
        Self(f16::from_f32(Default::default()))
    }
}

impl Clone for Half {
    fn clone(&self) -> Self {
        Half(self.0)
    }
}

impl Copy for Half {}

impl fmt::Debug for Half {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Half({:?})", self.0)
    }
}

unsafe impl Send for Half {}
unsafe impl Sync for Half {}
