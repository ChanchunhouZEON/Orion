/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;

/// Read a .fvecs file (binary format: [dim: u32, f32 * dim] per vector).
pub fn read_fvecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let dim = match reader.read_u32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let mut vec = vec![0.0f32; dim];
        reader.read_f32_into::<LittleEndian>(&mut vec)?;
        vectors.push(vec);
    }

    Ok(vectors)
}

/// Read a .ivecs file (binary format: [dim: u32, i32 * dim] per vector).
/// Returns as Vec<Vec<u32>> (neighbor indices are non-negative).
pub fn read_ivecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<u32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let dim = match reader.read_u32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let mut vec = vec![0i32; dim];
        for v in vec.iter_mut() {
            *v = reader.read_i32::<LittleEndian>()?;
        }
        vectors.push(vec.into_iter().map(|x| x as u32).collect());
    }

    Ok(vectors)
}

/// Write vectors in .fvecs format for testing.
#[cfg(test)]
fn write_fvecs<P: AsRef<Path>>(path: P, vectors: &[Vec<f32>]) -> io::Result<()> {
    use byteorder::WriteBytesExt;
    use std::io::BufWriter;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in vectors {
        writer.write_u32::<LittleEndian>(vec.len() as u32)?;
        for &v in vec {
            writer.write_f32::<LittleEndian>(v)?;
        }
    }
    Ok(())
}

/// Write vectors in .ivecs format for testing.
#[cfg(test)]
fn write_ivecs<P: AsRef<Path>>(path: P, vectors: &[Vec<u32>]) -> io::Result<()> {
    use byteorder::WriteBytesExt;
    use std::io::BufWriter;

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in vectors {
        writer.write_u32::<LittleEndian>(vec.len() as u32)?;
        for &v in vec {
            writer.write_i32::<LittleEndian>(v as i32)?;
        }
    }
    Ok(())
}

/// Read a .bvecs file (binary format: [dim: u8*4 as u32, u8 * dim] per vector).
/// Converts u8 to f32 for compatibility.
pub fn read_bvecs<P: AsRef<Path>>(path: P) -> io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let dim = match reader.read_u32::<LittleEndian>() {
            Ok(d) => d as usize,
            Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        };

        let mut bytes = vec![0u8; dim];
        reader.read_exact(&mut bytes)?;
        vectors.push(bytes.into_iter().map(|b| b as f32).collect());
    }

    Ok(vectors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fvecs_roundtrip() {
        let vectors = vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let tmp = std::env::temp_dir().join("test_fvecs.fvecs");
        write_fvecs(&tmp, &vectors).unwrap();

        let loaded = read_fvecs(&tmp).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded[1], vec![4.0, 5.0, 6.0]);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_ivecs_roundtrip() {
        let vectors = vec![vec![10u32, 20, 30], vec![40, 50, 60]];
        let tmp = std::env::temp_dir().join("test_ivecs.ivecs");
        write_ivecs(&tmp, &vectors).unwrap();

        let loaded = read_ivecs(&tmp).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0], vec![10, 20, 30]);
        assert_eq!(loaded[1], vec![40, 50, 60]);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_fvecs_empty_file() {
        let tmp = std::env::temp_dir().join("test_empty.fvecs");
        std::fs::write(&tmp, &[]).unwrap();

        let loaded = read_fvecs(&tmp).unwrap();
        assert!(loaded.is_empty());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_fvecs_single_vector() {
        let vectors = vec![vec![3.14f32]];
        let tmp = std::env::temp_dir().join("test_single.fvecs");
        write_fvecs(&tmp, &vectors).unwrap();

        let loaded = read_fvecs(&tmp).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!((loaded[0][0] - 3.14).abs() < 1e-6);

        std::fs::remove_file(&tmp).ok();
    }
}
