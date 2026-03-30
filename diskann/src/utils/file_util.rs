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
use std::path::Path;
use std::{io, mem};

use crate::model::data_store::DatasetDto;

pub fn load_metadata_from_file(file_name: &str) -> std::io::Result<(usize, usize)> {
    let file = File::open(file_name)?;
    let mut reader = BufReader::new(file);

    let npoints = reader.read_i32::<LittleEndian>()? as usize;
    let ndims = reader.read_i32::<LittleEndian>()? as usize;

    Ok((npoints, ndims))
}

pub fn load_ids_to_delete_from_file(file_name: &str) -> std::io::Result<(usize, Vec<u32>)> {
    let mut file = File::open(file_name)?;
    let num_ids = file.read_u32::<LittleEndian>()? as usize;

    let mut ids = Vec::with_capacity(num_ids);
    for _ in 0..num_ids {
        let id = file.read_u32::<LittleEndian>()?;
        ids.push(id);
    }

    Ok((num_ids, ids))
}

pub fn copy_aligned_data_from_file<T: Default + Copy>(
    bin_file: &str,
    dataset_dto: DatasetDto<T>,
    pts_offset: usize,
) -> std::io::Result<(usize, usize)> {
    let mut reader = File::open(bin_file)?;

    let npts = reader.read_i32::<LittleEndian>()? as usize;
    let dim = reader.read_i32::<LittleEndian>()? as usize;
    let rounded_dim = dataset_dto.rounded_dim;
    let offset = pts_offset * rounded_dim;

    for i in 0..npts {
        let data_slice =
            &mut dataset_dto.data[offset + i * rounded_dim..offset + i * rounded_dim + dim];
        let mut buf = vec![0u8; dim * mem::size_of::<T>()];
        reader.read_exact(&mut buf)?;

        let ptr = buf.as_ptr() as *const T;
        let temp_slice = unsafe { std::slice::from_raw_parts(ptr, dim) };
        data_slice.copy_from_slice(temp_slice);

        (i * rounded_dim + dim..i * rounded_dim + rounded_dim).for_each(|j| {
            dataset_dto.data[j] = T::default();
        });
    }

    Ok((npts, dim))
}

#[inline]
pub fn open_file_to_write(file_name: &str) -> std::io::Result<File> {
    OpenOptions::new()
        .write(true)
        .create(true)
        .open(Path::new(file_name))
}

pub fn delete_file(file_name: &str) -> std::io::Result<()> {
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
) -> std::io::Result<usize> {
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

pub fn load_bin<T: Copy>(
    bin_file: &str,
    file_offset: usize,
) -> std::io::Result<(Vec<T>, usize, usize)> {
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

pub fn get_file_size(filename: &str) -> io::Result<u64> {
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
        ) -> std::io::Result<usize> {
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
