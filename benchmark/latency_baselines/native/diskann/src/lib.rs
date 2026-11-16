/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Private benchmark bridge. One handle owns an already-built DiskANN index.
//! Python allocates result buffers inside its timed adapter call. Calls share the
//! read-only index/scratch pool; no build, query batching or subprocess per query.
use anyhow::{ensure, Result};
use orion_ann_diskann::{
    index::{create_inmem_index, ANNInmemIndex},
    model::{IndexConfiguration, IndexWriteParametersBuilder},
};
use orion_ann_vector::Metric;
use serde::Deserialize;
use std::{
    cell::RefCell,
    ffi::{c_char, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    path: String,
    count: usize,
    dimension: usize,
    degree: u32,
    build_l: u32,
    alpha: f32,
    threads: u32,
}
struct Handle {
    index: Box<dyn ANNInmemIndex<f32>>,
    dimension: usize,
    count: usize,
}
thread_local! { static ERROR: RefCell<CString> = RefCell::new(CString::default()); }
fn guarded<T>(fallback: T, f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        error => {
            let message = match error {
                Ok(Err(e)) => format!("{e:#}"),
                _ => "DiskANN panicked; discard this process/index".to_owned(),
            };
            ERROR.with(|e| *e.borrow_mut() = CString::new(message.replace('\0', " ")).unwrap());
            fallback
        }
    }
}
#[no_mangle]
extern "C" fn latency_error() -> *const c_char {
    ERROR.with(|e| e.borrow().as_ptr())
}

// SAFETY: the benchmark passes a live NUL-terminated UTF-8 JSON string. The
// returned handle must be destroyed once, after all concurrent searches finish.
#[no_mangle]
unsafe extern "C" fn latency_open(json: *const c_char) -> *mut Handle {
    guarded(std::ptr::null_mut(), || {
        ensure!(!json.is_null(), "null config");
        let c: Config = serde_json::from_str(CStr::from_ptr(json).to_str()?)?;
        ensure!(
            c.count > 0 && c.dimension > 0 && c.threads > 0,
            "invalid shape/threads"
        );
        let write = IndexWriteParametersBuilder::new(c.build_l, c.degree)
            .with_alpha(c.alpha)
            .with_num_threads(c.threads)
            .build();
        let config = IndexConfiguration::new(
            Metric::L2,
            c.dimension,
            c.dimension,
            c.count,
            false,
            0,
            false,
            0,
            1.0,
            write,
        );
        let mut index = create_inmem_index::<f32>(config)?;
        index.load(&c.path, c.count)?;
        Ok(Box::into_raw(Box::new(Handle {
            index,
            dimension: c.dimension,
            count: c.count,
        })))
    })
}
// SAFETY: handle is live; query has dimension f32 elements; output has k u32
// elements, exclusively owned by this call and non-overlapping with query.
#[no_mangle]
unsafe extern "C" fn latency_search(
    handle: *const Handle,
    query: *const f32,
    dimension: usize,
    k: usize,
    beam: u32,
    output: *mut u32,
) -> i32 {
    guarded(-1, || {
        ensure!(
            !handle.is_null() && !query.is_null() && !output.is_null(),
            "null buffer"
        );
        let h = &*handle;
        ensure!(
            dimension == h.dimension && k > 0 && k <= h.count && beam as usize >= k,
            "invalid search shape"
        );
        h.index.search(
            std::slice::from_raw_parts(query, dimension),
            k,
            beam,
            std::slice::from_raw_parts_mut(output, k),
        )?;
        Ok(0)
    })
}
// SAFETY: called once with the pointer from latency_open, without active calls.
#[no_mangle]
unsafe extern "C" fn latency_close(handle: *mut Handle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}
