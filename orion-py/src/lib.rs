/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Thin ownership/GIL boundary. Search behavior lives in orion-runtime.
#[cfg(panic = "abort")]
compile_error!("orion-py requires panic=unwind; build with --profile python");
mod errors;
use errors::{boundary, to_python, OrionError};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use orion_ann::{BaseInput, Index, IndexOptions};
use orion_ann::{Error, Result};
use orion_ann::{VectorFormat, VectorStorageKind};
use pyo3::prelude::*;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        RwLock,
    },
};

#[pyclass(name = "_Index", module = "orion_ann._native")]
struct PyIndex {
    inner: RwLock<Index>,
    invalidated: AtomicBool,
    dimension: usize,
    storage: VectorStorageKind,
}

fn options(json: &str) -> Result<IndexOptions> {
    serde_json::from_str(json).map_err(|e| {
        Error::invalid_argument(format!("invalid index options: {e}"))
            .at("configure")
            .with_source(e)
    })
}

fn array_base(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<BaseInput> {
    if let Ok(values) = array.extract::<PyReadonlyArray2<'_, f32>>() {
        if !values.is_c_contiguous() {
            return Err(to_python(
                py,
                Error::invalid_argument("base must be C-contiguous"),
            ));
        }
        return BaseInput::from_f32(values.as_slice()?, values.shape()[1])
            .map_err(|e| to_python(py, e));
    }
    if let Ok(values) = array.extract::<PyReadonlyArray2<'_, u8>>() {
        if !values.is_c_contiguous() {
            return Err(to_python(
                py,
                Error::invalid_argument("base must be C-contiguous"),
            ));
        }
        return BaseInput::from_u8(values.as_slice()?, values.shape()[1])
            .map_err(|e| to_python(py, e));
    }
    Err(to_python(
        py,
        Error::invalid_argument(
            "base must be a 2-D NumPy array with native float32 or uint8 dtype",
        ),
    ))
}

/// Query bytes/coordinates are owned before the GIL is released. A borrowed
/// NumPy view must never outlive the GIL while Rust workers can read it.
fn array_queries(
    py: Python<'_>,
    array: &Bound<'_, PyAny>,
    dimension: usize,
    storage: VectorStorageKind,
) -> PyResult<OwnedQueries> {
    if let Ok(values) = array.extract::<PyReadonlyArray2<'_, u8>>() {
        if values.shape()[1] != dimension {
            return Err(to_python(
                py,
                Error::dimension("queries.dimension", dimension, values.shape()[1])
                    .at("read_queries"),
            ));
        }
        if !values.is_c_contiguous() {
            return Err(to_python(
                py,
                Error::invalid_argument("queries must be C-contiguous"),
            ));
        }
        return Ok(OwnedQueries::U8(values.as_slice()?.to_vec()));
    }
    if storage == VectorStorageKind::F32 {
        if let Ok(values) = array.extract::<PyReadonlyArray2<'_, f32>>() {
            if values.shape()[1] != dimension {
                return Err(to_python(
                    py,
                    Error::dimension("queries.dimension", dimension, values.shape()[1])
                        .at("read_queries"),
                ));
            }
            if !values.is_c_contiguous() {
                return Err(to_python(
                    py,
                    Error::invalid_argument("queries must be C-contiguous"),
                ));
            }
            return Ok(OwnedQueries::F32(values.as_slice()?.to_vec()));
        }
    }
    Err(to_python(
        py,
        Error::invalid_argument(
            "queries must be a 2-D uint8 array for native-u8 storage, or float32/uint8 for f32 storage",
        ),
    ))
}

enum OwnedQueries {
    F32(Vec<f32>),
    U8(Vec<u8>),
}
impl OwnedQueries {
    fn batch(&self, dimension: usize) -> Result<orion_ann::QueryBatch<'_>> {
        match self {
            Self::F32(values) => orion_ann::QueryBatch::from_f32(values, dimension),
            Self::U8(values) => orion_ann::QueryBatch::from_u8(values, dimension),
        }
    }
    fn into_f32(self) -> Vec<f32> {
        match self {
            Self::F32(values) => values,
            Self::U8(values) => values.into_iter().map(f32::from).collect(),
        }
    }
}

enum Queries {
    Array(OwnedQueries),
    File(PathBuf, Option<VectorFormat>),
}
impl Queries {
    fn extract(
        py: Python<'_>,
        value: &Bound<'_, PyAny>,
        format: Option<&str>,
        dimension: usize,
        storage: VectorStorageKind,
    ) -> PyResult<Self> {
        if let Ok(path) = value.extract::<String>() {
            let format = format
                .map(|s| serde_json::from_value(serde_json::Value::String(s.into())))
                .transpose()
                .map_err(|e| {
                    to_python(
                        py,
                        Error::invalid_argument(format!("invalid query format: {e}")),
                    )
                })?;
            return Ok(Self::File(path.into(), format));
        }
        if format.is_some() {
            return Err(to_python(
                py,
                Error::invalid_argument("query_format applies only to files"),
            ));
        }
        Ok(Self::Array(array_queries(py, value, dimension, storage)?))
    }

    fn read(self, dimension: usize, storage: VectorStorageKind) -> Result<OwnedQueries> {
        match self {
            Self::Array(values) => Ok(values),
            Self::File(path, format) => {
                orion_ann::read_query_file(&path, format, dimension, storage).map(OwnedQueries::F32)
            }
        }
    }
}

// RwLock poisons exclusive writers only. A read-side backend panic must also
// invalidate the Python handle, matching the exception's reopen contract.
struct InvalidateOnUnwind<'a>(&'a AtomicBool);
impl Drop for InvalidateOnUnwind<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.store(true, Ordering::Release);
        }
    }
}

impl PyIndex {
    fn with_read<T>(&self, f: impl FnOnce(&Index) -> Result<T>) -> Result<T> {
        let index = self.inner.read().map_err(|_| errors::poisoned())?;
        if self.invalidated.load(Ordering::Acquire) {
            return Err(errors::poisoned());
        }
        let _guard = InvalidateOnUnwind(&self.invalidated);
        f(&index)
    }

    fn with_write<T>(&self, f: impl FnOnce(&mut Index) -> Result<T>) -> Result<T> {
        let mut index = self.inner.write().map_err(|_| errors::poisoned())?;
        if self.invalidated.load(Ordering::Acquire) {
            return Err(errors::poisoned());
        }
        f(&mut index)
    }

    fn wrap(index: Index) -> Self {
        Self {
            dimension: index.dimension(),
            storage: index.storage(),
            inner: RwLock::new(index),
            invalidated: AtomicBool::new(false),
        }
    }
}

#[pymethods]
impl PyIndex {
    #[cfg(feature = "test-hooks")]
    fn _test_panic(&self, py: Python<'_>) -> PyResult<()> {
        boundary(py, "test_panic", || {
            py.allow_threads(|| {
                let _guard = self.inner.write().unwrap();
                panic!("injected FFI boundary test");
            })
        })
    }

    #[cfg(feature = "test-hooks")]
    fn _test_read_panic(&self, py: Python<'_>) -> PyResult<()> {
        boundary(py, "test_read_panic", || {
            errors::result(
                py,
                py.allow_threads(|| {
                    self.with_read(|_| panic!("injected read-side FFI boundary test"))
                }),
            )
        })
    }

    #[staticmethod]
    fn from_file(py: Python<'_>, path: String, options_json: &str) -> PyResult<Self> {
        boundary(py, "new", || {
            let options = errors::result(py, options(options_json))?;
            py.allow_threads(move || Index::new(BaseInput::from_file(path), options))
                .map(Self::wrap)
                .map_err(|e| to_python(py, e))
        })
    }

    #[staticmethod]
    fn from_array(py: Python<'_>, base: &Bound<'_, PyAny>, options_json: &str) -> PyResult<Self> {
        boundary(py, "new", || {
            let options = errors::result(py, options(options_json))?;
            let input = array_base(py, base)?;
            py.allow_threads(move || Index::new(input, options))
                .map(Self::wrap)
                .map_err(|e| to_python(py, e))
        })
    }

    #[staticmethod]
    #[pyo3(signature = (reference, base, threads=None, memory_budget_gib=None))]
    fn open(
        py: Python<'_>,
        reference: String,
        base: &Bound<'_, PyAny>,
        threads: Option<usize>,
        memory_budget_gib: Option<f64>,
    ) -> PyResult<Self> {
        boundary(py, "open", || {
            let input = match base.extract::<String>() {
                Ok(p) => BaseInput::from_file(p),
                Err(_) => array_base(py, base)?,
            };
            py.allow_threads(move || {
                let source: orion_ann::SnapshotSource = serde_json::from_str(&reference)?;
                orion_ann::OpenOptions {
                    threads,
                    memory_budget_gib,
                }
                .open(source, input)
            })
            .map(Self::wrap)
            .map_err(|e| to_python(py, e))
        })
    }

    #[pyo3(signature = (queries, k, search_list_size=None, query_format=None))]
    fn calibrate(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyAny>,
        k: usize,
        search_list_size: Option<usize>,
        query_format: Option<&str>,
    ) -> PyResult<String> {
        boundary(py, "calibrate", || {
            let queries =
                Queries::extract(py, queries, query_format, self.dimension, self.storage)?;
            py.allow_threads(move || {
                let values = queries.read(self.dimension, self.storage)?;
                // Lock only after releasing the GIL. Otherwise a second Python
                // thread can block the first thread from reacquiring it on return.
                self.with_write(|index| {
                    serde_json::to_string(&index.calibrate(
                        values.into_f32(),
                        k,
                        search_list_size,
                    )?)
                    .map_err(Error::from)
                })
            })
            .map_err(|e| to_python(py, e))
        })
    }

    #[pyo3(signature = (queries, k, search_list_size, *, query_format=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
        search_list_size: usize,
        query_format: Option<&str>,
    ) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>)> {
        boundary(py, "search", || {
            let queries =
                Queries::extract(py, queries, query_format, self.dimension, self.storage)?;
            let result = py
                .allow_threads(move || {
                    let values = queries.read(self.dimension, self.storage)?;
                    self.with_read(|index| {
                        index.search(
                            values.batch(self.dimension)?,
                            &orion_ann::SearchOptions::new(k, search_list_size),
                        )
                    })
                })
                .map_err(|e| to_python(py, e))?;
            let ids = ndarray::Array2::from_shape_vec((result.counts.len(), result.k), result.ids)
                .map_err(|e| to_python(py, Error::internal(e.to_string()).with_source(e)))?;
            Ok((
                ids.into_pyarray_bound(py),
                result.counts.into_pyarray_bound(py),
            ))
        })
    }

    fn save(&self, py: Python<'_>, path: String) -> PyResult<String> {
        boundary(py, "save", || {
            py.allow_threads(move || {
                self.with_read(|index| {
                    let reference = index.save(std::path::Path::new(&path))?;
                    serde_json::to_string(&reference).map_err(Error::from)
                })
            })
            .map_err(|e| to_python(py, e))
        })
    }

    fn info(&self, py: Python<'_>) -> PyResult<String> {
        boundary(py, "info", || {
            py.allow_threads(|| {
                self.with_read(|index| serde_json::to_string(&index.info()).map_err(Error::from))
            })
            .map_err(|e| to_python(py, e))
        })
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add("OrionError", m.py().get_type_bound::<OrionError>())?;
    Ok(())
}
