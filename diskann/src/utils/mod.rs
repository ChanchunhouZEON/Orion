pub mod file_util;
pub use file_util::*;

#[allow(clippy::module_inception)]
pub mod utils;
pub use utils::*;

pub mod bit_vec_extension;
pub use bit_vec_extension::*;

pub mod rayon_util;
pub use rayon_util::*;

pub mod timer;
pub use timer::*;

pub mod cached_reader;
pub use cached_reader::*;

pub mod cached_writer;
pub use cached_writer::*;

pub mod kmeans;
pub use kmeans::*;

pub mod math_util;
pub use math_util::*;

pub mod partition;
pub use partition::*;

/// Return current process RSS as a human-readable string (macOS / Linux).
pub fn mem_usage() -> String {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let pid = std::process::id();
        if let Ok(out) = Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output() {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(kb) = s.trim().parse::<u64>() {
                    let mb = kb as f64 / 1024.0;
                    return format!("{:.1} MB (RSS)", mb);
                }
            }
        }
        "unknown".to_string()
    }
    #[cfg(not(target_os = "macos"))]
    {
        "unsupported".to_string()
    }
}
