/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Decode a fixture through the same length-delimited stream used in production.
use orion_ann_core::Result;
use std::{fs::File, io::Read, path::Path};

pub fn decode<T>(
    path: impl AsRef<Path>,
    decode: impl FnOnce(&mut dyn Read, u64) -> Result<T>,
) -> Result<T> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    decode(&mut file, len)
}
