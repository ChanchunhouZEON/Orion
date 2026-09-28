// Minimal PyO3 binding for Orion — SIFT 1M only (dim=128, L2,
// cascade = no prefilter + L2U8 admission + F32 rerank). Used as a
// calibration probe: comparing this binding's QPS against the Rust
// `benchmark` binary's QPS on the same cached index isolates the
// per-batch FFI overhead, which is the same overhead USearch /
// hnswlib see in their own Python bindings. Apples-to-apples.

use std::path::PathBuf;

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use diskann::model::InmemDataset;
use orion::algorithm::search::stage::admission::L2U8Admission;
use orion::algorithm::search::stage::rerank::F32Rerank;
use orion::algorithm::search::stage::PrefilterStage;
use orion::Orion;
use std::hint::black_box;

const DIM: usize = 128;

#[pyclass]
struct OrionSift {
    inner: Orion<DIM>,
    epsilon: f32,
    early_exit_limit: usize,
    window_size: usize,
}

#[pymethods]
impl OrionSift {
    /// Load a cached Orion index (`.bin` + `.pgraph` + `.qds`
    /// produced by the Rust benchmark / orion_sweep). `base` is the
    /// raw f32 base matrix the cache was built from — the cache
    /// itself only carries graph + sidecars, not the f32 base data.
    #[staticmethod]
    fn load_cache(cache_path: String, base: PyReadonlyArray2<f32>) -> PyResult<Self> {
        let shape = base.shape();
        if shape.len() != 2 || shape[1] != DIM {
            return Err(PyValueError::new_err(format!(
                "base must be (n, {DIM}); got shape {:?}",
                shape
            )));
        }
        let n = shape[0];
        let slice = base.as_slice()?;

        let empty_ds = InmemDataset::<f32, DIM>::new(0, 1.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut orion =
            Orion::<DIM>::load_from_cache(&PathBuf::from(&cache_path), empty_ds)
                .map_err(|e| PyRuntimeError::new_err(format!("load_from_cache: {e}")))?;

        let mut ds = InmemDataset::<f32, DIM>::new(n, 1.0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        ds.data
            .memcpy(slice)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        orion.dataset = ds;

        Ok(Self {
            inner: orion,
            epsilon: 0.0,
            early_exit_limit: usize::MAX,
            window_size: 8,
        })
    }

    /// Run Orion's auto-calibration at the given search-list
    /// size + window size, store the resulting (epsilon, early_exit)
    /// so subsequent `search_batch` calls use them. Returns the pair
    /// for inspection.
    fn calibrate(
        &mut self,
        warmup: PyReadonlyArray2<f32>,
        search_list_size: usize,
        window_size: usize,
    ) -> PyResult<(f32, usize)> {
        let shape = warmup.shape();
        if shape.len() != 2 || shape[1] != DIM {
            return Err(PyValueError::new_err(format!(
                "warmup must be (n, {DIM}); got {:?}",
                shape
            )));
        }
        let slice = warmup.as_slice()?;
        let n = shape[0];
        let mut qarrs: Vec<[f32; DIM]> = Vec::with_capacity(n);
        for i in 0..n {
            let mut a = [0.0f32; DIM];
            a.copy_from_slice(&slice[i * DIM..(i + 1) * DIM]);
            qarrs.push(a);
        }
        let calib = self
            .inner
            .calibrate(&qarrs, search_list_size, window_size, Default::default())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        self.epsilon = calib.threshold;
        self.early_exit_limit = calib.early_exit_limit;
        self.window_size = window_size;
        Ok((self.epsilon, self.early_exit_limit))
    }

    /// Batch search. Mirrors the single-FFI-call shape of USearch's
    /// `index.search(queries, count=k, threads=t)` — entire batch
    /// goes into Rust in one call, search runs under the rayon
    /// global pool (controlled via `RAYON_NUM_THREADS`), result
    /// comes back as one (n, k) numpy u32 array.
    fn search_batch<'py>(
        &self,
        py: Python<'py>,
        queries: PyReadonlyArray2<f32>,
        k: usize,
        search_list_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<u32>>> {
        let shape = queries.shape();
        if shape.len() != 2 || shape[1] != DIM {
            return Err(PyValueError::new_err(format!(
                "queries must be (n, {DIM}); got {:?}",
                shape
            )));
        }
        let n = shape[0];
        let slice = queries.as_slice()?;

        let mut qarrs: Vec<[f32; DIM]> = Vec::with_capacity(n);
        for i in 0..n {
            let mut a = [0.0f32; DIM];
            a.copy_from_slice(&slice[i * DIM..(i + 1) * DIM]);
            qarrs.push(a);
        }

        let admission = L2U8Admission::new(self.inner.ensure_quantized_dataset());
        let rerank = F32Rerank::new(&self.inner.dataset);
        let ws = self.window_size;
        let eps = self.epsilon;
        let ee = self.early_exit_limit;

        // Drop the GIL during the parallel search.
        let results: Vec<Vec<u32>> = py.allow_threads(|| {
            let no_pf: Option<&dyn PrefilterStage<DIM>> = None;
            self.inner
                .search_batch_unified(
                    &qarrs,
                    k,
                    search_list_size,
                    ws,
                    eps,
                    ee,
                    no_pf,
                    &admission,
                    &rerank,
                )
                .unwrap_or_default()
        });

        let mut out = Array2::<u32>::from_elem((n, k), u32::MAX);
        for (i, row) in results.iter().enumerate() {
            for (j, &id) in row.iter().take(k).enumerate() {
                out[[i, j]] = id;
            }
        }
        Ok(out.into_pyarray_bound(py))
    }

    /// Expose the calibrated (epsilon, early_exit_limit) for logging.
    #[getter]
    fn calibrated(&self) -> (f32, usize) {
        (self.epsilon, self.early_exit_limit)
    }

    /// rayon global pool size — handy for sanity-checking that
    /// RAYON_NUM_THREADS took effect across the FFI boundary.
    #[staticmethod]
    fn rayon_threads() -> usize {
        rayon::current_num_threads()
    }
}

/// ParlayANN-style 40 MB cache eviction. Bit-for-bit identical to
/// `benchmark::utils::flush_cache` so each timed trial starts from
/// the same cold-cache state as the Rust binary's sweep. Routed
/// through the Python binding so methodology is symmetric: any
/// Python-side QPS reading is comparable to the Rust JSON's, with
/// no warm-cache or per-binding cache-state asymmetry.
#[pyfunction]
fn flush_cache() {
    const N: usize = 5_000_000;
    let mut v: Vec<u64> = (0..N as u64).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..N).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        v.swap(i, j);
    }
    black_box(&v);
}

#[pymodule]
fn orion_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OrionSift>()?;
    m.add_function(wrap_pyfunction!(flush_cache, m)?)?;
    Ok(())
}
