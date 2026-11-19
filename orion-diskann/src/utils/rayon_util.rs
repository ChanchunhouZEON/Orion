/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::ops::Range;

use crate::common::ANNResult;

/// Executes a range serially for one requested worker, otherwise on the active Rayon pool.
///
/// The parallel branch does not create a pool or enforce an exact thread count.
/// Any callback failure is returned to the caller.
#[inline]
pub fn execute_with_rayon<F>(range: Range<usize>, num_threads: u32, f: F) -> ANNResult<()>
where
    F: Fn(usize) -> ANNResult<()> + Sync + Send + Copy,
{
    if num_threads == 1 {
        for i in range {
            f(i)?;
        }
        Ok(())
    } else {
        range.into_par_iter().try_for_each(f)
    }
}

/// Sets the process environment default read by a subsequently initialized Rayon pool.
///
/// This does not reconfigure existing pools. Use only during single-threaded
/// startup before concurrent environment access.
#[inline]
pub fn set_rayon_num_threads(num_threads: u32) {
    unsafe { std::env::set_var("RAYON_NUM_THREADS", num_threads.to_string()) };
}
