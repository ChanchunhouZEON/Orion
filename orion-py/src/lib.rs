//! Thin ownership/GIL boundary. Search behavior lives in orion-runtime.
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use orion_runtime::{
    config::VectorStorageKind,
    data::query::{QuerySource, VectorReader},
    data::VectorFormat,
};
use orion_runtime::{BaseInput, Index, IndexOptions};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
};
use std::{io::BufReader, path::PathBuf, sync::Mutex};

pyo3::create_exception!(_native, OrionError, PyRuntimeError);
fn runtime_error(message: String) -> PyErr {
    OrionError::new_err(message)
}

#[pyclass(name = "_Index", module = "orion_py._native")]
struct PyIndex {
    inner: Mutex<Index>,
    dimension: usize,
    storage: VectorStorageKind,
}

fn options(json: &str) -> PyResult<IndexOptions> {
    serde_json::from_str(json)
        .map_err(|e| PyValueError::new_err(format!("invalid index options: {e}")))
}

fn array_base(array: &Bound<'_, PyAny>) -> PyResult<BaseInput> {
    if let Ok(values) = array.extract::<PyReadonlyArray2<'_, f32>>() {
        if !values.is_c_contiguous() {
            return Err(PyValueError::new_err("base must be C-contiguous"));
        }
        return BaseInput::from_f32(values.as_slice()?, values.shape()[1]).map_err(runtime_error);
    }
    if let Ok(values) = array.extract::<PyReadonlyArray2<'_, u8>>() {
        if !values.is_c_contiguous() {
            return Err(PyValueError::new_err("base must be C-contiguous"));
        }
        return BaseInput::from_u8(values.as_slice()?, values.shape()[1]).map_err(runtime_error);
    }
    Err(PyTypeError::new_err(
        "base must be a 2-D NumPy array with native float32 or uint8 dtype",
    ))
}

/// Query bytes/coordinates are owned before the GIL is released. A borrowed
/// NumPy view must never outlive the GIL while Rust workers can read it.
fn array_queries(
    array: &Bound<'_, PyAny>,
    dimension: usize,
    storage: VectorStorageKind,
) -> PyResult<Vec<f32>> {
    if let Ok(values) = array.extract::<PyReadonlyArray2<'_, u8>>() {
        if values.shape()[1] != dimension || !values.is_c_contiguous() {
            return Err(PyValueError::new_err(
                "query shape or C-contiguity does not match index",
            ));
        }
        return Ok(values.as_slice()?.iter().map(|&v| f32::from(v)).collect());
    }
    if storage == VectorStorageKind::F32 {
        if let Ok(values) = array.extract::<PyReadonlyArray2<'_, f32>>() {
            if values.shape()[1] != dimension || !values.is_c_contiguous() {
                return Err(PyValueError::new_err(
                    "query shape or C-contiguity does not match index",
                ));
            }
            return Ok(values.as_slice()?.to_vec());
        }
    }
    Err(PyTypeError::new_err(
        "queries must be a 2-D uint8 array for native-u8 storage, or float32/uint8 for f32 storage",
    ))
}

enum Queries {
    Array(Vec<f32>),
    File(PathBuf, Option<VectorFormat>),
}
impl Queries {
    fn extract(
        value: &Bound<'_, PyAny>,
        format: Option<&str>,
        dimension: usize,
        storage: VectorStorageKind,
    ) -> PyResult<Self> {
        if let Ok(path) = value.extract::<String>() {
            let format = format
                .map(|s| serde_json::from_value(serde_json::Value::String(s.into())))
                .transpose()
                .map_err(|e| PyValueError::new_err(format!("invalid query format: {e}")))?;
            return Ok(Self::File(path.into(), format));
        }
        if format.is_some() {
            return Err(PyValueError::new_err("query_format applies only to files"));
        }
        Ok(Self::Array(array_queries(value, dimension, storage)?))
    }

    fn read(self, dimension: usize, storage: VectorStorageKind) -> Result<Vec<f32>, String> {
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
                let file = std::fs::File::open(&path)
                    .map_err(|e| format!("queries {}: {e}", path.display()))?;
                // Decode without normalization; the runtime normalizes once,
                // identically for file and ndarray queries.
                let mut reader = VectorReader::new(
                    BufReader::new(file),
                    format,
                    dimension,
                    orion_runtime::search::cascade::SearchMetric::L2,
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
    #[staticmethod]
    fn from_file(py: Python<'_>, path: String, options_json: &str) -> PyResult<Self> {
        let options = options(options_json)?;
        py.allow_threads(move || Index::new(BaseInput::File(path.into()), options))
            .map(Self::wrap)
            .map_err(runtime_error)
    }

    #[staticmethod]
    fn from_array(py: Python<'_>, base: &Bound<'_, PyAny>, options_json: &str) -> PyResult<Self> {
        let options = options(options_json)?;
        let input = array_base(base)?;
        py.allow_threads(move || Index::new(input, options))
            .map(Self::wrap)
            .map_err(runtime_error)
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
        let input = match base.extract::<String>() {
            Ok(p) => BaseInput::File(p.into()),
            Err(_) => array_base(base)?,
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
        .map_err(runtime_error)
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
        let queries = Queries::extract(queries, query_format, self.dimension, self.storage)?;
        py.allow_threads(move || {
            let values = queries.read(self.dimension, self.storage)?;
            // Lock only after releasing the GIL. Otherwise a second Python
            // thread can block the first thread from reacquiring it on return.
            let mut index = self
                .inner
                .lock()
                .map_err(|_| "index operation panicked; reopen the index".to_string())?;
            serde_json::to_string(&index.calibrate(values, k, search_list_size)?)
                .map_err(|e| e.to_string())
        })
        .map_err(runtime_error)
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
        let queries = Queries::extract(queries, query_format, self.dimension, self.storage)?;
        let result = py
            .allow_threads(move || {
                let values = queries.read(self.dimension, self.storage)?;
                self.inner
                    .lock()
                    .map_err(|_| "index operation panicked; reopen the index".to_string())?
                    .search(values, k, search_list_size, adaptive)
            })
            .map_err(runtime_error)?;
        let ids = ndarray::Array2::from_shape_vec((result.counts.len(), result.k), result.ids)
            .map_err(|e| runtime_error(e.to_string()))?;
        Ok((
            ids.into_pyarray_bound(py),
            result.counts.into_pyarray_bound(py),
        ))
    }

    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        py.allow_threads(move || {
            self.inner
                .lock()
                .map_err(|_| "index lock poisoned".to_string())?
                .save(std::path::Path::new(&path))
        })
        .map_err(runtime_error)
    }

    fn info(&self, py: Python<'_>) -> PyResult<String> {
        py.allow_threads(|| {
            let index = self
                .inner
                .lock()
                .map_err(|_| "index lock poisoned".to_string())?;
            serde_json::to_string(&index.info()).map_err(|e| e.to_string())
        })
        .map_err(runtime_error)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIndex>()?;
    m.add("OrionError", m.py().get_type_bound::<OrionError>())?;
    Ok(())
}
