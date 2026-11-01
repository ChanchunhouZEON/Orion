//! Thin ownership/GIL boundary. Search behavior lives in orion-runtime.
#[cfg(panic = "abort")]
compile_error!("orion-py requires panic=unwind; build with --profile python");
mod errors;
use errors::{boundary, to_python, OrionError};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use orion_ann_runtime::{
    config::VectorStorageKind,
    data::query::{QuerySource, VectorReader},
    data::VectorFormat,
};
use orion_ann_runtime::{BaseInput, Index, IndexOptions};
use orion_ann_runtime::{Error, Result};
use pyo3::prelude::*;
use std::{io::BufReader, path::PathBuf, sync::Mutex};

#[pyclass(name = "_Index", module = "orion_ann._native")]
struct PyIndex {
    inner: Mutex<Index>,
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
) -> PyResult<Vec<f32>> {
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
        return Ok(values.as_slice()?.iter().map(|&v| f32::from(v)).collect());
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
            return Ok(values.as_slice()?.to_vec());
        }
    }
    Err(to_python(
        py,
        Error::invalid_argument(
            "queries must be a 2-D uint8 array for native-u8 storage, or float32/uint8 for f32 storage",
        ),
    ))
}

enum Queries {
    Array(Vec<f32>),
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

    fn read(self, dimension: usize, storage: VectorStorageKind) -> Result<Vec<f32>> {
        match self {
            Self::Array(values) => Ok(values),
            Self::File(path, format) => {
                let format = match format {
                    Some(f) => f,
                    None => VectorFormat::infer(&path)?,
                };
                if storage == VectorStorageKind::U8
                    && !matches!(format, VectorFormat::Bvecs | VectorFormat::U8bin)
                {
                    return Err("native u8 queries require bvecs/u8bin input".into());
                }
                let file = std::fs::File::open(&path).map_err(|e| Error::io("open", &path, e))?;
                // Decode without normalization; the runtime normalizes once,
                // identically for file and ndarray queries.
                let mut reader = VectorReader::new(
                    BufReader::new(file),
                    format,
                    dimension,
                    orion_ann_runtime::search::cascade::SearchMetric::L2,
                )?;
                let mut values = Vec::new();
                let mut batch = Vec::new();
                while reader.read_batch(&mut batch, 1024)? != 0 {
                    values.extend_from_slice(&batch);
                }
                Ok(values)
            }
        }
    }
}

impl PyIndex {
    fn wrap(index: Index) -> Self {
        Self {
            dimension: index.dimension(),
            storage: index.storage(),
            inner: Mutex::new(index),
        }
    }
}

#[pymethods]
impl PyIndex {
    #[cfg(feature = "test-hooks")]
    fn _test_panic(&self, py: Python<'_>) -> PyResult<()> {
        boundary(py, "test_panic", || {
            py.allow_threads(|| {
                let _guard = self.inner.lock().unwrap();
                panic!("injected FFI boundary test");
            })
        })
    }

    #[staticmethod]
    fn from_file(py: Python<'_>, path: String, options_json: &str) -> PyResult<Self> {
        boundary(py, "new", || {
            let options = errors::result(py, options(options_json))?;
            py.allow_threads(move || Index::new(BaseInput::File(path.into()), options))
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
    #[pyo3(signature = (path, base, threads=None, memory_budget_gib=None))]
    fn open(
        py: Python<'_>,
        path: String,
        base: &Bound<'_, PyAny>,
        threads: Option<usize>,
        memory_budget_gib: Option<f64>,
    ) -> PyResult<Self> {
        boundary(py, "open", || {
            let input = match base.extract::<String>() {
                Ok(p) => BaseInput::File(p.into()),
                Err(_) => array_base(py, base)?,
            };
            py.allow_threads(move || {
                Index::open(
                    std::path::Path::new(&path),
                    input,
                    threads,
                    memory_budget_gib,
                )
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
                let mut index = self.inner.lock().map_err(|_| errors::poisoned())?;
                serde_json::to_string(&index.calibrate(values, k, search_list_size)?)
                    .map_err(Error::from)
            })
            .map_err(|e| to_python(py, e))
        })
    }

    #[pyo3(signature = (queries, k, search_list_size, adaptive=true, query_format=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
        search_list_size: usize,
        adaptive: bool,
        query_format: Option<&str>,
    ) -> PyResult<(Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<u32>>)> {
        boundary(py, "search", || {
            let queries =
                Queries::extract(py, queries, query_format, self.dimension, self.storage)?;
            let result = py
                .allow_threads(move || {
                    let values = queries.read(self.dimension, self.storage)?;
                    self.inner.lock().map_err(|_| errors::poisoned())?.search(
                        values,
                        k,
                        search_list_size,
                        adaptive,
                    )
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

    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        boundary(py, "save", || {
            py.allow_threads(move || {
                self.inner
                    .lock()
                    .map_err(|_| errors::poisoned())?
                    .save(std::path::Path::new(&path))
            })
            .map_err(|e| to_python(py, e))
        })
    }

    fn info(&self, py: Python<'_>) -> PyResult<String> {
        boundary(py, "info", || {
            py.allow_threads(|| {
                let index = self.inner.lock().map_err(|_| errors::poisoned())?;
                serde_json::to_string(&index.info()).map_err(Error::from)
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
