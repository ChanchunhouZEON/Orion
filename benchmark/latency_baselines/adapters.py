# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Per-engine read-only search adapters. Search knobs change only between trials.

Python indices are built only with --prepare-missing and committed with matching
metadata. Native adapters only load existing graphs. No search implementation is
replaced by a different algorithm, and missing dependencies are never substituted.
"""
import ctypes as ct
import json
from pathlib import Path

import numpy as np

from .data import identity, normalized, rows, write_json

ENGINES = (
    "orion",
    "parlayann",
    "diskann",
    "hnswlib",
    "usearch",
    "lancedb",
    "faiss-hnsw-sq8",
    "faiss-ivf-flat",
    "faiss-ivf-pq",
    "annoy",
)


class Unsupported(ValueError):
    pass


class Adapter:
    def initialize_thread(self):
        pass

    def close(self):
        pass


class Native(Adapter):
    def __init__(self, engine, config, dataset, base, threads):
        self.engine, self.metric = engine, dataset["metric"]
        self.unit_norm_ip = config.get("unit_norm_ip", False)
        if self.metric == "squared-l2" and config.get("normalize", False):
            raise Unsupported("normalizing squared-L2 would change the search metric")
        if self.metric == "inner-product" and (
            self.unit_norm_ip or config.get("normalize", False)
        ):
            for start in range(0, len(base), 8192):
                norms = np.linalg.norm(
                    np.asarray(base[start : start + 8192], dtype=np.float64), axis=1
                )
                if not np.allclose(norms, 1.0, rtol=2e-6, atol=2e-6):
                    raise Unsupported(
                        "IP-to-normalized search requires an already unit-norm base"
                    )
        library = Path(config["library"]).resolve()
        if not library.is_file():
            raise FileNotFoundError(
                f"{library}; build adapters with latency_baselines.build_native"
            )
        self.lib = ct.CDLL(str(library))  # CDLL releases the GIL during the native call
        self.lib.latency_error.restype = ct.c_char_p
        self.lib.latency_close.argtypes = [ct.c_void_p]
        self.lib.latency_close.restype = None
        self.lib.latency_search.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_float),
            ct.c_size_t,
            ct.c_size_t,
            ct.c_uint32,
            ct.POINTER(ct.c_uint32),
        ]
        self.lib.latency_search.restype = ct.c_int
        self.lib.latency_open.restype = ct.c_void_p
        self.handle = None
        if engine == "parlayann":
            source, graph = Path(config["base_fbin"]), Path(config["graph"])
            native_base = rows(source, "fbin")
            if len(native_base) != len(base):
                raise ValueError("ParlayANN fbin/base count mismatch")
            self.native_dimension = native_base.shape[1]
            normalize = self.metric == "cosine" or config.get("normalize", False)
            # Converted PA data must represent the same logical vectors. Check
            # in bounded blocks, without keeping another full dataset in RAM.
            for start in range(0, len(base), 8192):
                actual, expected = (
                    native_base[start : start + 8192],
                    base[start : start + 8192],
                )
                if normalize:
                    actual, expected = normalized(actual), normalized(expected)
                width = min(actual.shape[1], expected.shape[1])
                equal = (
                    np.allclose(
                        actual[:, :width], expected[:, :width], rtol=1e-6, atol=1e-7
                    )
                    if normalize
                    else np.array_equal(actual[:, :width], expected[:, :width])
                )
                if (
                    not equal
                    or np.any(actual[:, width:] != 0)
                    or np.any(expected[:, width:] != 0)
                ):
                    raise ValueError(
                        "ParlayANN fbin differs from the configured base, beyond zero padding/declared normalization"
                    )
            if not graph.is_file():
                raise FileNotFoundError(graph)
            bits, mode = config["quantize_bits"], config["quantize_mode"]
            if bits not in (0, 8, 16) or mode not in (
                (0, 1, 3) if self.metric == "squared-l2" else (0, 1, 5)
            ):
                raise Unsupported("unsupported ParlayANN quantization recipe")
            self.lib.latency_thread_init.argtypes = []
            self.lib.latency_thread_init.restype = None
            self.lib.latency_open.argtypes = [
                ct.c_char_p,
                ct.c_char_p,
                ct.c_int,
                ct.c_int,
                ct.c_int,
                ct.c_double,
            ]
            self.handle = self.lib.latency_open(
                str(source).encode(),
                str(graph).encode(),
                2 if normalize else {"squared-l2": 0, "inner-product": 1}[self.metric],
                bits,
                mode,
                config.get("rerank_factor", 2),
            )
        else:
            if self.metric == "inner-product" and not self.unit_norm_ip:
                raise Unsupported(
                    "DiskANN adapter uses L2; raw inner product is not equivalent"
                )
            path = Path(config["index"])
            for file in (path, Path(str(path) + ".data")):
                if not file.is_file():
                    raise FileNotFoundError(file)
            manifest_path = path.with_suffix(".manifest.json")
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
            elif config.get("build"):
                manifest = dict(
                    config["build"],
                    count=len(base),
                    metric="l2",
                    storage="f32",
                    dimension=rows(str(path) + ".data", "fbin").shape[1],
                )
            else:
                raise ValueError(
                    "legacy DiskANN cache requires explicit degree/build_l/alpha"
                )
            saved = rows(str(path) + ".data", "fbin")
            self.native_dimension = saved.shape[1]
            if (
                len(saved) != len(base)
                or saved.shape[1] < base.shape[1]
                or manifest.get("count") != len(base)
                or manifest.get("dimension") != saved.shape[1]
                or manifest.get("metric") != "l2"
                or manifest.get("storage") != "f32"
            ):
                raise ValueError("DiskANN cache manifest/shape mismatch")
            # Old experiments used separate normalized/padded base files. Verify
            # their actual cached coordinates, not a guessed filename identity.
            # This is outside timing and does not allocate a second full base.
            for start in range(0, len(base), 8192):
                expected = np.asarray(base[start : start + 8192], dtype=np.float32)
                if self.metric == "cosine" or self.unit_norm_ip:
                    expected = normalized(expected)
                actual = saved[start : start + 8192]
                if not np.allclose(
                    actual[:, : base.shape[1]], expected, rtol=1e-6, atol=1e-7
                ) or np.any(actual[:, base.shape[1] :] != 0):
                    raise ValueError(
                        "DiskANN cached coordinates differ from configured base/metric"
                    )
            args = dict(
                path=str(path),
                count=len(base),
                dimension=saved.shape[1],
                threads=threads,
                **{k: manifest[k] for k in ("degree", "build_l", "alpha")},
            )
            self.lib.latency_open.argtypes = [ct.c_char_p]
            self.handle = self.lib.latency_open(json.dumps(args).encode())
        if not self.handle:
            raise RuntimeError(self.lib.latency_error().decode())

    def initialize_thread(self):
        if self.engine == "parlayann":
            self.lib.latency_thread_init()

    def configure(self, effort, k):
        if not k <= effort <= np.iinfo(np.uint32).max:
            raise ValueError("native beam must be >= k and fit u32")
        self.effort, self.k = effort, k

    def search(self, query):
        q = (
            normalized(query[None, :])[0]
            if self.engine == "diskann"
            and (self.metric == "cosine" or self.unit_norm_ip)
            else query
        )
        if q.size < self.native_dimension:
            q = np.pad(q, (0, self.native_dimension - q.size))
        elif q.size > self.native_dimension:
            if np.any(q[self.native_dimension :] != 0):
                raise ValueError(
                    "query contains nonzero coordinates outside cached dimensions"
                )
            q = q[: self.native_dimension]
        output = np.full(self.k, np.iinfo(np.uint32).max, dtype=np.uint32)
        if (
            self.lib.latency_search(
                self.handle,
                q.ctypes.data_as(ct.POINTER(ct.c_float)),
                q.size,
                self.k,
                self.effort,
                output.ctypes.data_as(ct.POINTER(ct.c_uint32)),
            )
            != 0
        ):
            raise RuntimeError(self.lib.latency_error().decode())
        return output[output != np.iinfo(np.uint32).max]

    def close(self):
        if self.handle:
            self.lib.latency_close(self.handle)
            self.handle = None


class Orion(Adapter):
    def __init__(self, config, dataset, queries, threads):
        from orion_ann import Index, GraphConfig, CascadeConfig, StorageOptions

        if config.get("snapshot"):
            self.index = Index.open(
                config["snapshot"],
                base=dataset["base"],
                threads=threads,
                auto_pin=config.get("auto_pin", True),
            )
        elif config.get("experiment_config"):
            source_path = Path(config["experiment_config"])
            source = json.loads(source_path.read_text())
            options = dict(source["index"])
            graph = options.pop("graph")
            if options.get("cache_dir"):
                options["cache_dir"] = str(
                    (source_path.parent / options["cache_dir"]).resolve()
                )
            if graph.get("staged_file"):
                graph["staged_file"] = str(
                    (source_path.parent / graph["staged_file"]).resolve()
                )
            if graph.get("source") != "parlayann" or not graph.get("staged_file"):
                raise ValueError(
                    "Orion reuse requires an explicit existing staged graph"
                )
            if not Path(graph["staged_file"]).is_file():
                raise FileNotFoundError(graph["staged_file"])
            options["metric"] = dataset["metric"]
            options["threads"] = threads
            options["graph"] = GraphConfig(**graph)
            options["cascade"] = CascadeConfig(**options["cascade"])
            if "storage_options" in options:
                options["storage_options"] = StorageOptions(
                    **options["storage_options"]
                )
            self.index = Index.from_file(dataset["base"], **options)
        else:
            raise ValueError(
                "Orion requires snapshot or experiment_config with explicit staged graph"
            )
        info = self.index.info
        if callable(info):
            info = info()
        if info["metric"] != dataset["metric"]:
            raise ValueError(
                "Orion snapshot metric does not match the comparison metric"
            )
        self.info = info
        self.queries = queries
        self.config = config

    def configure(self, effort, k):
        if effort < k:
            raise ValueError("Orion beam must be >= k")
        if k not in self.index.calibrated_ks:
            self.index.calibrate(
                self.queries[: self.config.get("calibration_samples", 200)],
                k=k,
                search_list_size=self.config.get("calibration_l", max(200, k)),
            )
        self.k, self.effort = k, effort

    def search(self, query):
        result = self.index.search(
            query[None, :], k=self.k, search_list_size=self.effort
        )
        return result.ids[0, : int(result.counts[0])]


class PythonIndex(Adapter):
    def __init__(self, engine, config, dataset, base, threads, prepare):
        self.engine, self.metric, self.config = engine, dataset["metric"], config
        self.dimension, self.count = base.shape[1], len(base)
        self.cache = Path(config["cache"])
        metadata = self.cache.with_name(self.cache.name + ".latency.json")
        build = config.get("build", {})
        stamp = dict(engine=engine, base=identity(dataset, base), build=build)
        exists = self.cache.exists()
        if exists:
            if not metadata.is_file() or json.loads(metadata.read_text()) != stamp:
                raise ValueError(
                    f"{self.cache}: missing/mismatched latency cache identity; use a new path"
                )
        elif not prepare:
            raise FileNotFoundError(
                f"{self.cache}: use --prepare-missing to build explicitly"
            )
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self._open(base, build, threads, exists)
        if not exists:
            write_json(
                metadata, stamp
            )  # a failed build never publishes a cache identity

    def _open(self, base, build, threads, exists):
        e, metric, d, n = self.engine, self.metric, self.dimension, self.count
        if e.startswith("faiss-"):
            import faiss

            self.faiss = faiss
            faiss.omp_set_num_threads(threads)
            if exists:
                self.index = faiss.read_index(str(self.cache))
            else:
                metric_id = (
                    faiss.METRIC_L2
                    if metric == "squared-l2"
                    else faiss.METRIC_INNER_PRODUCT
                )
                values = (
                    normalized(base)
                    if metric == "cosine"
                    else np.ascontiguousarray(base, dtype=np.float32)
                )
                if e == "faiss-hnsw-sq8":
                    self.index = faiss.IndexHNSWSQ(
                        d, faiss.ScalarQuantizer.QT_8bit, build.get("m", 16), metric_id
                    )
                    self.index.hnsw.efConstruction = build.get("ef_construction", 200)
                else:
                    nlist = build.get("nlist", min(256, max(1, n // 40)))
                    quantizer = faiss.IndexFlat(d, metric_id)
                    if e == "faiss-ivf-flat":
                        self.index = faiss.IndexIVFFlat(quantizer, d, nlist, metric_id)
                    else:
                        m = build.get(
                            "pq_m",
                            max(v for v in range(1, max(1, d // 8) + 1) if d % v == 0),
                        )
                        if d % m:
                            raise Unsupported(
                                "FAISS PQ dimension must be divisible by pq_m"
                            )
                        self.index = faiss.IndexIVFPQ(
                            quantizer, d, nlist, m, build.get("pq_bits", 8), metric_id
                        )
                self.index.train(values)
                self.index.add(values)
                faiss.write_index(self.index, str(self.cache))
            if self.index.ntotal != n or self.index.d != d:
                raise ValueError("FAISS cache shape mismatch")
        elif e == "hnswlib":
            import hnswlib

            self.index = hnswlib.Index(
                space={"squared-l2": "l2", "inner-product": "ip", "cosine": "cosine"}[
                    metric
                ],
                dim=d,
            )
            if exists:
                self.index.load_index(str(self.cache), max_elements=n)
            else:
                self.index.init_index(
                    max_elements=n,
                    ef_construction=build.get("ef_construction", 200),
                    M=build.get("m", 16),
                    random_seed=build.get("seed", 100),
                )
                self.index.add_items(base, np.arange(n), num_threads=threads)
                self.index.save_index(str(self.cache))
            self.index.set_num_threads(1)
            if self.index.get_current_count() != n:
                raise ValueError("hnswlib cache count mismatch")
        elif e == "usearch":
            from usearch.index import Index, MetricKind

            if exists:
                self.index = Index.restore(str(self.cache), view=False)
            else:
                self.index = Index(
                    ndim=d,
                    metric={
                        "squared-l2": MetricKind.L2sq,
                        "cosine": MetricKind.Cos,
                        "inner-product": MetricKind.IP,
                    }[metric],
                    dtype="f32",
                    connectivity=build.get("m", 16),
                    expansion_add=build.get("ef_construction", 200),
                )
                self.index.add(
                    np.arange(n, dtype=np.uint64),
                    np.ascontiguousarray(base, dtype=np.float32),
                    threads=threads,
                )
                self.index.save(str(self.cache))
            if self.index is None or len(self.index) != n or self.index.ndim != d:
                raise ValueError("USearch cache shape mismatch")
        elif e == "annoy":
            from annoy import AnnoyIndex

            self.index = AnnoyIndex(
                d,
                {
                    "squared-l2": "euclidean",
                    "cosine": "angular",
                    "inner-product": "dot",
                }[metric],
            )
            if exists:
                if not self.index.load(str(self.cache), prefault=True):
                    raise ValueError("Annoy cache load failed")
            else:
                self.index.set_seed(build.get("seed", 100))
                for i, row in enumerate(base):
                    self.index.add_item(i, row)
                self.index.build(build.get("trees", 50), n_jobs=threads)
                self.index.save(str(self.cache))
            if self.index.get_n_items() != n:
                raise ValueError("Annoy cache count mismatch")
        elif e == "lancedb":
            import lancedb

            self.db = lancedb.connect(str(self.cache))
            if exists:
                self.index = self.db.open_table("vectors")
            else:
                import pyarrow as pa

                values = pa.FixedSizeListArray.from_arrays(
                    pa.array(np.asarray(base, dtype=np.float32).ravel()), d
                )
                self.index = self.db.create_table(
                    "vectors",
                    pa.table({"id": np.arange(n, dtype=np.int64), "vec": values}),
                )
                self.index.create_index(
                    metric={
                        "squared-l2": "L2",
                        "cosine": "cosine",
                        "inner-product": "dot",
                    }[metric],
                    vector_column_name="vec",
                    index_type="IVF_HNSW_SQ",
                    num_partitions=build.get("partitions", 1),
                    m=build.get("m", 16),
                    ef_construction=build.get("ef_construction", 200),
                )
            if self.index.count_rows() != n:
                raise ValueError("LanceDB cache count mismatch")
        else:
            raise Unsupported(e)

    def initialize_thread(self):
        if self.engine.startswith("faiss-"):
            self.faiss.omp_set_num_threads(1)  # OpenMP's setting is thread-local

    def configure(self, effort, k):
        self.effort, self.k = effort, k
        if (
            self.engine in ("hnswlib", "usearch", "lancedb", "faiss-hnsw-sq8")
            and effort < k
        ):
            raise ValueError("graph search effort must be >= k")
        if self.engine == "hnswlib":
            self.index.set_ef(effort)
        elif self.engine == "usearch":
            self.index.expansion_search = effort
        elif self.engine == "faiss-hnsw-sq8":
            self.index.hnsw.efSearch = effort
        elif self.engine.startswith("faiss-ivf"):
            if not 1 <= effort <= self.index.nlist:
                raise ValueError("nprobe must be in 1..nlist")
            self.index.nprobe = effort
        elif effort <= 0:
            raise ValueError("search effort must be positive")

    def search(self, query):
        e = self.engine
        if e.startswith("faiss-"):
            q = (
                normalized(query[None, :])
                if self.metric == "cosine"
                else query[None, :]
            )
            _, ids = self.index.search(q, self.k)
            return ids[0, ids[0] >= 0]  # IVF can legitimately return fewer than k
        if e == "hnswlib":
            return self.index.knn_query(query[None, :], k=self.k, num_threads=1)[0][0]
        if e == "usearch":
            return self.index.search(query, count=self.k, threads=1).keys
        if e == "annoy":
            return self.index.get_nns_by_vector(
                query.tolist(), self.k, search_k=self.effort
            )
        return [
            r["id"]
            for r in self.index.search(query.tolist(), vector_column_name="vec")
            .limit(self.k)
            .ef(self.effort)
            .nprobes(self.config.get("nprobes", 1))
            .select(["id"])
            .to_list()
        ]


def open_adapter(engine, config, dataset, base, queries, threads, prepare):
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine}")
    if engine in ("parlayann", "diskann"):
        return Native(engine, config, dataset, base, threads)
    if engine == "orion":
        return Orion(config, dataset, queries, threads)
    return PythonIndex(engine, config, dataset, base, threads, prepare)
