/*
 * Copyright (c) Chanchunhou. All rights reserved.
 * Licensed under the MIT License.
 */

//! Native public search binding. Both the historical private entry and the
//! descriptor use one implementation so A/B tests change only the Python frame.
use super::*;

#[doc = include_str!("search_contract.md")]
#[pyclass(name = "_SearchDescriptor", module = "orion_ann._native", frozen)]
pub(super) struct SearchDescriptor;

#[pymethods]
impl SearchDescriptor {
    /// Creates the stateless descriptor used to bind public search directly to native code.
    #[new]
    fn new() -> Self {
        Self
    }

    /// Binds instance search to its native handle while preserving subclass method lookup.
    fn __get__(
        slf: PyRef<'_, Self>,
        instance: Option<&Bound<'_, PyAny>>,
        _owner: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let py = slf.py();
        match instance {
            // Resolving a normal method already creates a bound method. Keep
            // that behavior, including instance/subclass overrides; no Python
            // wrapper or per-instance monkeypatch is installed here.
            Some(instance) if !instance.is_none() => Ok(instance
                .getattr(intern!(py, "_handle"))?
                .getattr(intern!(py, "_search_public"))?
                .unbind()),
            _ => Ok(slf.into_py(py)),
        }
    }

    /// Builds the public introspection signature without adding work to ordinary searches.
    #[getter]
    fn __signature__(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Preserve the Python facade's unbound self keyword as well as the
        // bound method's defaults. This runs only during introspection.
        let inspect = py.import_bound("inspect")?;
        let signature = inspect.call_method1(
            "signature",
            (py.get_type_bound::<PyIndex>().getattr("_search_public")?,),
        )?;
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item(
            "kind",
            inspect
                .getattr("Parameter")?
                .getattr("POSITIONAL_OR_KEYWORD")?,
        )?;
        let mut parameters = Vec::new();
        for (i, parameter) in signature
            .getattr("parameters")?
            .call_method0("values")?
            .iter()?
            .enumerate()
        {
            let parameter = parameter?;
            parameters.push(if i == 0 {
                parameter
                    .call_method("replace", (), Some(&kwargs))?
                    .unbind()
            } else {
                parameter.unbind()
            });
        }
        kwargs.clear();
        kwargs.set_item("parameters", parameters)?;
        Ok(signature
            .call_method("replace", (), Some(&kwargs))?
            .unbind())
    }

    /// Supports unbound class-level search calls while preserving Python self-argument errors.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, pyo3::types::PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<PyObject> {
        // Class-level Index.search(index, ...) is uncommon but remains valid,
        // including Index.search(self=index, queries=...). Instance calls bind
        // the native method directly and do not enter this parsing path.
        let keywords = match kwargs {
            Some(kwargs) => kwargs.copy()?,
            None => PyDict::new_bound(py),
        };
        let keyword_self = keywords.get_item("self")?;
        let instance = if !args.is_empty() {
            if keyword_self.is_some() {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "search() got multiple values for argument 'self'",
                ));
            }
            args.get_item(0)?
        } else {
            keyword_self.ok_or_else(|| {
                pyo3::exceptions::PyTypeError::new_err("search() missing required argument 'self'")
            })?
        };
        if keywords.contains("self")? {
            keywords.del_item("self")?;
        }
        let arguments =
            pyo3::types::PyTuple::new_bound(py, args.iter().skip(usize::from(!args.is_empty())));
        Ok(instance
            .getattr(intern!(py, "_handle"))?
            .getattr(intern!(py, "_search_public"))?
            .call(arguments, Some(&keywords))?
            .unbind())
    }
}

impl PyIndex {
    /// Own all query temporaries inside this function, so their drops happen
    /// before the timestamp that marks the end of GIL-free native execution.
    fn execute_queries(
        &self,
        queries: Queries,
        k: usize,
        beam: usize,
        scores: bool,
    ) -> Result<orion_ann::SearchResult> {
        execute(
            &self.inner,
            &self.invalidated,
            queries,
            self.dimension,
            self.storage,
            k,
            beam,
            scores,
        )
    }

    /// Copies query input, runs native search without the GIL, then constructs the Python result.
    ///
    /// Optional probes distinguish native execution, GIL return and result assembly;
    /// normal execution uses the same search body without recording timestamps.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn search_impl<'py>(
        &self,
        py: Python<'py>,
        queries: &Bound<'py, PyAny>,
        k: usize,
        beam: usize,
        query_format: Option<&str>,
        scores: bool,
        result_type: Option<&Bound<'py, PyType>>,
    ) -> PyResult<PyObject> {
        boundary(py, "search", || {
            #[cfg(feature = "search-timing")]
            let started = std::time::Instant::now();
            let queries =
                Queries::extract(py, queries, query_format, self.dimension, self.storage)?;
            #[cfg(feature = "search-timing")]
            let before_release = std::time::Instant::now();

            #[cfg(feature = "search-timing")]
            let (result, native_start, native_end) = py.allow_threads(move || {
                let native_start = std::time::Instant::now();
                let result = self.execute_queries(queries, k, beam, scores);
                (result, native_start, std::time::Instant::now())
            });
            #[cfg(not(feature = "search-timing"))]
            let result = py.allow_threads(move || self.execute_queries(queries, k, beam, scores));

            // This first instruction after allow_threads measures the return
            // from native completion through GIL reacquisition (including OS
            // scheduling/guard overhead, not an isolated mutex-wait counter).
            #[cfg(feature = "search-timing")]
            let gil_returned = std::time::Instant::now();
            let result = result.map_err(|e| to_python(py, e))?;
            #[cfg(feature = "search-timing")]
            let wrapping_start = std::time::Instant::now();
            let result = wrap_search_output(py, search_output(py, result)?, result_type)?;
            #[cfg(feature = "search-timing")]
            {
                let done = std::time::Instant::now();
                search_timing::record((
                    search_timing::ns(before_release - started),
                    search_timing::ns(native_start - before_release),
                    search_timing::ns(native_end - native_start),
                    search_timing::ns(gil_returned - native_end),
                    search_timing::ns(done - wrapping_start),
                    search_timing::ns(done - started),
                ));
            }
            Ok(result)
        })
    }
}

/// Shared calculation body: async changes ownership and delivery only.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute(
    inner: &RwLock<Index>,
    invalidated: &AtomicBool,
    queries: Queries,
    dimension: usize,
    storage: VectorStorageKind,
    k: usize,
    beam: usize,
    scores: bool,
) -> Result<orion_ann::SearchResult> {
    let values = queries.read(dimension, storage)?;
    let index = inner.read().map_err(|_| errors::poisoned())?;
    if invalidated.load(Ordering::Acquire) {
        return Err(errors::poisoned());
    }
    let _guard = InvalidateOnUnwind(invalidated);
    index.search(
        orion_ann::QueryBatch::from_input(values.as_input(), dimension)?,
        &orion_ann::SearchOptions::new(k, beam).with_scores(scores),
    )
}
