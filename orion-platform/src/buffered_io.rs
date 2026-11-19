/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Bounded sequential I/O shared by dataset and graph codecs.

use std::io::{BufWriter, Read, Write};

/// Default buffer size for sequential dataset and graph writes.
pub const FILE_WRITE_BYTES: usize = 8 * 1024 * 1024;

/// Maximum destination span passed to a sequential payload reader.
/// This bounds individual read requests, not total RSS or filesystem cache usage.
pub const FILE_READ_BYTES: usize = 8 * 1024 * 1024;

/// Completed mapped-output window between optional writeback/reclaim checkpoints.
/// This is an I/O scheduling threshold, not a reservation or a bound on OS RSS.
pub const RECLAIM_WINDOW_BYTES: usize = 32 * 1024 * 1024;

/// Fill a payload in bounded blocks without allocating a second payload buffer.
///
/// A `BufReader` can bypass its internal buffer for large destinations. Explicit
/// slicing keeps the requested spans bounded even when the destination is mmap.
/// The byte stream and caller-owned destination layout remain unchanged.
///
/// # Errors
/// Propagates I/O failures and premature EOF; an error can leave a filled prefix.
/// Interrupted and short reads follow [`Read::read_exact`] semantics.
pub fn read_exact_in_blocks<R: Read + ?Sized>(
    reader: &mut R,
    bytes: &mut [u8],
) -> std::io::Result<()> {
    for block in bytes.chunks_mut(FILE_READ_BYTES) {
        reader.read_exact(block)?;
    }
    Ok(())
}

/// Fill the current block before allowing BufWriter to bypass its buffer.
/// A small header followed by a large slice would otherwise flush the header
/// alone. On the tested Lustre system, small writes inflated disk allocation.
///
/// Uses the supplied writer's capacity as the block size. The caller must flush
/// the final partial block and handle any flush error before publishing a file.
/// This helper does not change the byte stream or interpret its file format.
pub fn write_in_blocks<W: Write>(
    writer: &mut BufWriter<W>,
    mut bytes: &[u8],
) -> std::io::Result<()> {
    if writer.capacity() == 0 {
        return writer.write_all(bytes);
    }
    while !bytes.is_empty() {
        if writer.buffer().len() == writer.capacity() {
            writer.flush()?;
        }
        let count = bytes.len().min(writer.capacity() - writer.buffer().len());
        writer.write_all(&bytes[..count])?;
        bytes = &bytes[count..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_writer_combines_header_and_payload_and_propagates_errors() {
        #[derive(Default)]
        struct Sink {
            sizes: Vec<usize>,
            bytes: Vec<u8>,
            fail: bool,
        }
        impl Write for Sink {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if self.fail {
                    return Err(std::io::ErrorKind::StorageFull.into());
                }
                self.sizes.push(bytes.len());
                self.bytes.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for header_len in [8, 16] {
            let mut output = BufWriter::with_capacity(FILE_WRITE_BYTES, Sink::default());
            output.write_all(&vec![7; header_len]).unwrap();
            let payload = vec![19; FILE_WRITE_BYTES * 2 + 128 - header_len];
            write_in_blocks(&mut output, &payload).unwrap();
            output.flush().unwrap();
            assert_eq!(
                output.get_ref().sizes,
                [FILE_WRITE_BYTES, FILE_WRITE_BYTES, 128]
            );
            assert_eq!(&output.get_ref().bytes[..header_len], vec![7; header_len]);
            assert_eq!(&output.get_ref().bytes[header_len..], payload);
        }
        let mut output = BufWriter::with_capacity(
            8,
            Sink {
                fail: true,
                ..Sink::default()
            },
        );
        assert!(write_in_blocks(&mut output, &[1; 16]).is_err());
    }
}

#[cfg(test)]
mod read_tests {
    use super::*;

    struct Source {
        left: usize,
        largest: usize,
        interrupt: bool,
    }
    impl Read for Source {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            self.largest = self.largest.max(out.len());
            if std::mem::take(&mut self.interrupt) {
                return Err(std::io::ErrorKind::Interrupted.into());
            }
            let n = self.left.min(out.len()).min(1024 * 1024 + 7);
            out[..n].fill(37);
            self.left -= n;
            Ok(n)
        }
    }

    #[test]
    fn bounded_reads_handle_headers_short_reads_interrupts_and_tails() {
        let size = FILE_READ_BYTES * 2 + 31;
        let source = Source {
            left: size + 16,
            largest: 0,
            interrupt: true,
        };
        let mut reader = std::io::BufReader::with_capacity(FILE_READ_BYTES, source);
        reader.read_exact(&mut [0; 16]).unwrap();
        let mut out = vec![0; size];
        read_exact_in_blocks(&mut reader, &mut out).unwrap();
        assert!(out.iter().all(|&v| v == 37));
        assert!(reader.get_ref().largest <= FILE_READ_BYTES);
        read_exact_in_blocks(&mut reader, &mut []).unwrap();
        assert_eq!(
            read_exact_in_blocks(&mut reader, &mut [0])
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::UnexpectedEof
        );
    }
}
