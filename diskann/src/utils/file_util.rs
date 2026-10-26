/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::mem;
use std::path::Path;

use crate::common::ANNResult;
use crate::model::data_store::DatasetDto;

pub fn load_metadata_from_file(file_name: &str) -> ANNResult<(usize, usize)> {
    let file = File::open(file_name)?;
    let mut reader = BufReader::new(file);

    let npoints = reader.read_i32::<LittleEndian>()? as usize;
    let ndims = reader.read_i32::<LittleEndian>()? as usize;

    Ok((npoints, ndims))
}

pub fn load_ids_to_delete_from_file(file_name: &str) -> ANNResult<(usize, Vec<u32>)> {
    let mut file = File::open(file_name)?;
    let num_ids = file.read_u32::<LittleEndian>()? as usize;

    let mut ids = Vec::with_capacity(num_ids);
    for _ in 0..num_ids {
        let id = file.read_u32::<LittleEndian>()?;
        ids.push(id);
    }

    Ok((num_ids, ids))
}

// Keep each IO request bounded even when the resident dataset is hundreds of GB.
const DATASET_READ_BYTES: usize = 8 * 1024 * 1024;

/// Read packed coordinates into their final aligned rows. Pod guarantees that
/// every input bit pattern is valid, including a partially filled buffer on IO
/// failure; no unaligned byte-buffer-to-T cast is needed.
pub fn copy_aligned_data_from_file<T: Default + bytemuck::Pod>(
    bin_file: &str,
    dataset_dto: DatasetDto<T>,
    pts_offset: usize,
) -> ANNResult<(usize, usize)> {
    use std::io::{Error, ErrorKind};
    let invalid = |message| Error::new(ErrorKind::InvalidData, message);
    let mut reader = File::open(bin_file)?;
    let npts = reader.read_i32::<LittleEndian>()?;
    let dim = reader.read_i32::<LittleEndian>()?;
    if npts < 0 || dim <= 0 || dim as usize > dataset_dto.rounded_dim {
        return Err(invalid("invalid dataset count/dimension or destination stride").into());
    }
    let (npts, dim) = (npts as usize, dim as usize);
    let element_bytes = mem::size_of::<T>();
    if element_bytes == 0 || (!cfg!(target_endian = "little") && element_bytes > 1) {
        return Err(invalid("binary dataset requires nonzero little-endian elements").into());
    }
    let expected_bytes = (npts as u64)
        .checked_mul(dim as u64)
        .and_then(|n| n.checked_mul(element_bytes as u64))
        .and_then(|n| n.checked_add(8))
        .ok_or_else(|| invalid("dataset file size overflow"))?;
    if reader.metadata()?.len() != expected_bytes {
        return Err(invalid("dataset header/file length mismatch").into());
    }
    let stride = dataset_dto.rounded_dim;
    let start = pts_offset
        .checked_mul(stride)
        .ok_or_else(|| invalid("dataset offset overflow"))?;
    let end = npts
        .checked_mul(stride)
        .and_then(|n| start.checked_add(n))
        .ok_or_else(|| invalid("dataset destination size overflow"))?;
    let destination = dataset_dto
        .data
        .get_mut(start..end)
        .ok_or_else(|| invalid("dataset exceeds destination capacity"))?;
    read_aligned_payload(&mut reader, destination, dim, stride)?;
    Ok((npts, dim))
}

fn read_aligned_payload<R: Read, T: Default + bytemuck::Pod>(
    reader: &mut R,
    destination: &mut [T],
    dim: usize,
    stride: usize,
) -> std::io::Result<()> {
    if dim == stride {
        // SIFT128/f32 and native u8 land directly in resident storage. There is
        // no staging allocation or extra memcpy, including the final short block.
        for block in bytemuck::cast_slice_mut::<T, u8>(destination).chunks_mut(DATASET_READ_BYTES) {
            reader.read_exact(block)?;
        }
    } else {
        // Padded dimensions require unpacking. Reuse one block for the entire
        // file rather than allocating and reading a temporary for every vector.
        if destination.is_empty() {
            return Ok(());
        }
        let rows_per_block = (DATASET_READ_BYTES / mem::size_of::<T>() / dim)
            .max(1)
            .min(destination.len() / stride);
        let mut packed = vec![T::default(); rows_per_block * dim];
        for block in destination.chunks_mut(rows_per_block * stride) {
            let rows = block.len() / stride;
            let packed = &mut packed[..rows * dim];
            reader.read_exact(bytemuck::cast_slice_mut(packed))?;
            for (row, values) in block.chunks_exact_mut(stride).zip(packed.chunks_exact(dim)) {
                row[..dim].copy_from_slice(values);
                row[dim..].fill(T::default());
            }
        }
    }
    Ok(())
}

#[inline]
pub fn open_file_to_write(file_name: &str) -> ANNResult<File> {
    Ok(OpenOptions::new()
        .write(true)
        .create(true)
        .open(Path::new(file_name))?)
}

pub fn delete_file(file_name: &str) -> ANNResult<()> {
    if file_exists(file_name) {
        fs::remove_file(file_name)?;
    }
    Ok(())
}

pub fn file_exists(filename: &str) -> bool {
    std::path::Path::new(filename).exists()
}

pub fn save_data_in_base_dimensions<T: Default + Copy>(
    filename: &str,
    data: &mut [T],
    npts: usize,
    ndims: usize,
    aligned_dim: usize,
    offset: usize,
) -> ANNResult<usize> {
    let mut writer = open_file_to_write(filename)?;
    let npts_i32 = npts as i32;
    let ndims_i32 = ndims as i32;
    let bytes_written = 2 * std::mem::size_of::<u32>() + npts * ndims * (std::mem::size_of::<T>());

    writer.seek(std::io::SeekFrom::Start(offset as u64))?;
    writer.write_all(&npts_i32.to_le_bytes())?;
    writer.write_all(&ndims_i32.to_le_bytes())?;
    let data_ptr = data.as_ptr() as *const u8;
    for i in 0..npts {
        let middle_offset = i * aligned_dim * std::mem::size_of::<T>();
        let middle_slice = unsafe {
            std::slice::from_raw_parts(
                data_ptr.add(middle_offset),
                ndims * std::mem::size_of::<T>(),
            )
        };
        writer.write_all(middle_slice)?;
    }
    writer.flush()?;
    Ok(bytes_written)
}

pub fn load_bin<T: Copy>(bin_file: &str, file_offset: usize) -> ANNResult<(Vec<T>, usize, usize)> {
    let mut reader = File::open(bin_file)?;
    reader.seek(std::io::SeekFrom::Start(file_offset as u64))?;
    let npts = reader.read_i32::<LittleEndian>()? as usize;
    let dim = reader.read_i32::<LittleEndian>()? as usize;

    let size = npts * dim * std::mem::size_of::<T>();
    let mut buf = vec![0u8; size];
    reader.read_exact(&mut buf)?;

    let ptr = buf.as_ptr() as *const T;
    let data = unsafe { std::slice::from_raw_parts(ptr, npts * dim) };

    Ok((data.to_vec(), npts, dim))
}

pub fn get_file_size(filename: &str) -> ANNResult<u64> {
    let reader = File::open(filename)?;
    let metadata = reader.metadata()?;
    Ok(metadata.len())
}

macro_rules! save_bin {
    ($name:ident, $t:ty, $write_func:ident) => {
        pub fn $name(
            filename: &str,
            data: &[$t],
            num_pts: usize,
            dims: usize,
            offset: usize,
        ) -> ANNResult<usize> {
            let mut writer = open_file_to_write(filename)?;
            println!("Writing bin: {}", filename);
            writer.seek(SeekFrom::Start(offset as u64))?;
            let num_pts_i32 = num_pts as i32;
            let dims_i32 = dims as i32;
            let bytes_written = num_pts * dims * mem::size_of::<$t>() + 2 * mem::size_of::<u32>();
            writer.write_i32::<LittleEndian>(num_pts_i32)?;
            writer.write_i32::<LittleEndian>(dims_i32)?;
            println!(
                "bin: #pts = {}, #dims = {}, size = {}B",
                num_pts, dims, bytes_written
            );
            for item in data.iter() {
                writer.$write_func::<LittleEndian>(*item)?;
            }
            writer.flush()?;
            println!("Finished writing bin.");
            Ok(bytes_written)
        }
    };
}

save_bin!(save_bin_f32, f32, write_f32);
save_bin!(save_bin_u64, u64, write_u64);
save_bin!(save_bin_u32, u32, write_u32);

#[cfg(test)]
mod dataset_read_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Input(std::path::PathBuf);
    impl Input {
        fn new(count: i32, dim: i32, payload: &[u8]) -> Self {
            let path = std::env::temp_dir().join(format!(
                "diskann-block-load-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            let mut file = File::create(&path).unwrap();
            file.write_all(&count.to_le_bytes()).unwrap();
            file.write_all(&dim.to_le_bytes()).unwrap();
            file.write_all(payload).unwrap();
            Self(path)
        }
        fn load<T: Default + bytemuck::Pod>(
            &self,
            data: &mut [T],
            stride: usize,
            offset: usize,
        ) -> ANNResult<(usize, usize)> {
            copy_aligned_data_from_file(
                self.0.to_str().unwrap(),
                DatasetDto {
                    data,
                    rounded_dim: stride,
                },
                offset,
            )
        }
    }
    impl Drop for Input {
        fn drop(&mut self) {
            fs::remove_file(&self.0).unwrap();
        }
    }

    #[test]
    fn packed_f32_crosses_block_boundary_and_preserves_surrounding_storage() {
        let rows = DATASET_READ_BYTES / (128 * 4) + 3;
        let values: Vec<f32> = (0..rows * 128).map(|i| (i % 997) as f32 - 498.5).collect();
        let file = Input::new(rows as i32, 128, bytemuck::cast_slice(&values));
        let mut output = vec![-999.; (rows + 2) * 128 + 16];
        assert_eq!(file.load(&mut output, 128, 2).unwrap(), (rows, 128));
        assert_eq!(&output[..256], &[-999.; 256]);
        assert_eq!(&output[256..256 + values.len()], values.as_slice());
        assert_eq!(&output[256 + values.len()..], &[-999.; 16]);
    }

    #[test]
    fn padded_rows_cross_block_boundary_and_zero_padding_at_append_offset() {
        let rows = DATASET_READ_BYTES / (3 * 4) + 2;
        let values: Vec<f32> = (0..rows * 3).map(|i| (i % 101) as f32).collect();
        let file = Input::new(rows as i32, 3, bytemuck::cast_slice(&values));
        let mut output = vec![-1.; (rows + 2) * 4];
        file.load(&mut output, 4, 1).unwrap();
        assert_eq!(&output[..4], &[-1.; 4]);
        assert_eq!(&output[output.len() - 4..], &[-1.; 4]);
        for (row, expected) in output[4..4 + rows * 4]
            .chunks_exact(4)
            .zip(values.chunks_exact(3))
        {
            assert_eq!(&row[..3], expected);
            assert_eq!(row[3], 0.);
        }
    }

    #[test]
    fn byte_and_half_coordinates_preserve_bits() {
        let file = Input::new(2, 3, &[0, 128, 255, 19, 37, 91]);
        let mut output = [77u8; 8];
        file.load(&mut output, 4, 0).unwrap();
        assert_eq!(output, [0, 128, 255, 0, 19, 37, 91, 0]);
        let values = [half::f16::from_f32(-1.5), half::f16::from_f32(0.25)];
        let file = Input::new(1, 2, bytemuck::cast_slice(&values));
        let mut output = [half::f16::ZERO; 2];
        file.load(&mut output, 2, 0).unwrap();
        assert_eq!(output, values);
    }

    #[test]
    fn malformed_shape_length_capacity_and_offset_fail_before_writing() {
        for (count, dim, payload, stride, offset) in [
            (-1, 2, vec![], 2, 0),
            (1, 0, vec![], 2, 0),
            (1, 3, vec![1, 2, 3], 2, 0),
            (2, 2, vec![1, 2, 3], 2, 0), // truncated
            (1, 2, vec![1, 2, 3], 2, 0), // trailing byte
            (3, 2, vec![1; 6], 2, 0),    // destination too small
            (1, 2, vec![1, 2], 2, usize::MAX),
        ] {
            let file = Input::new(count, dim, &payload);
            let mut output = [99u8; 4];
            assert!(file.load(&mut output, stride, offset).is_err());
            assert_eq!(output, [99; 4]);
        }
    }

    #[test]
    fn packed_reader_uses_block_requests_and_propagates_short_reads() {
        struct Requests(Vec<usize>);
        impl Read for Requests {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                self.0.push(buffer.len());
                buffer.fill(13);
                Ok(buffer.len())
            }
        }
        let mut output = vec![0u8; DATASET_READ_BYTES + 128];
        let mut reader = Requests(Vec::new());
        read_aligned_payload(&mut reader, &mut output, 128, 128).unwrap();
        assert_eq!(reader.0, [DATASET_READ_BYTES, 128]);
        assert!(output.iter().all(|v| *v == 13));
        let mut truncated = std::io::Cursor::new([1u8; 127]);
        assert_eq!(
            read_aligned_payload(&mut truncated, &mut output[..128], 128, 128)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }
}
