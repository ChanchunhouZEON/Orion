pub mod aligned_reader;
pub use aligned_reader::*;

#[cfg(target_os = "windows")]
pub mod windows_aligned_file_reader;
#[cfg(target_os = "windows")]
pub use windows_aligned_file_reader::*;

#[cfg(target_family = "unix")]
pub mod unix_aligned_file_reader;
#[cfg(target_family = "unix")]
pub use unix_aligned_file_reader::*;
