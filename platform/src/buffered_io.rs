/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Buffered sequential writes shared by dataset and graph serializers.

use std::io::{BufWriter, Write};

/// Default buffer size for sequential dataset and graph writes.
pub const FILE_WRITE_BYTES: usize = 8 * 1024 * 1024;

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
