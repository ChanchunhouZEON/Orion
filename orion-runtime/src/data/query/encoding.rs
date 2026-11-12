/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! File payload decoding only. Shape/EOF checks live in the reader; coordinate
//! validation and normalization belong to shared query preparation.
use crate::{Error, QueryBuffer, Result, VectorFormat};
pub(super) fn empty(format: VectorFormat) -> QueryBuffer {
    match format {
        VectorFormat::Fvecs | VectorFormat::Fbin => QueryBuffer::F32(Vec::new()),
        VectorFormat::Bvecs | VectorFormat::U8bin => QueryBuffer::U8(Vec::new()),
        VectorFormat::I8bin => QueryBuffer::I8(Vec::new()),
        VectorFormat::U16bin => QueryBuffer::U16(Vec::new()),
        VectorFormat::U32bin => QueryBuffer::U32(Vec::new()),
    }
}
pub(super) fn reset(output: &mut QueryBuffer, format: VectorFormat) {
    let expected = empty(format);
    if std::mem::discriminant(output) != std::mem::discriminant(&expected) {
        *output = expected;
    }
    output.clear();
}
pub(super) fn append(output: &mut QueryBuffer, bytes: &[u8]) {
    match output {
        QueryBuffer::F32(v) => v.extend(
            bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap())),
        ),
        QueryBuffer::U8(v) => v.extend_from_slice(bytes),
        QueryBuffer::I8(v) => v.extend(bytes.iter().map(|&x| x as i8)),
        QueryBuffer::U16(v) => v.extend(
            bytes
                .chunks_exact(2)
                .map(|x| u16::from_le_bytes(x.try_into().unwrap())),
        ),
        QueryBuffer::U32(v) => v.extend(
            bytes
                .chunks_exact(4)
                .map(|x| u32::from_le_bytes(x.try_into().unwrap())),
        ),
    }
}
/// Compatibility transport cannot silently discard wide integer coordinates.
pub(super) fn legacy_f32(input: &QueryBuffer) -> Result<Vec<f32>> {
    if let QueryBuffer::U32(v) = input {
        if v.iter().any(|&x| f64::from(x as f32) != f64::from(x)) {
            return Err(Error::invalid_argument(
                "wide u32 queries require typed query transport",
            ));
        }
    }
    Ok(input.as_input().to_f32().into_owned())
}
