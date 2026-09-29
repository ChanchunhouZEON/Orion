"""Little-endian Yael vector writers shared by data preparation scripts."""
from pathlib import Path
import numpy as np


def _write_vecs(path, vectors, dtype, chunk_rows=16384):
    path = Path(path)
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or vectors.shape[1] <= 0:
        raise ValueError("Expected a two-dimensional vector array")
    path.parent.mkdir(parents=True, exist_ok=True)
    dimension = vectors.shape[1]
    with path.open("wb") as stream:
        for start in range(0, len(vectors), chunk_rows):
            chunk = vectors[start:start + chunk_rows]
            records = np.empty((len(chunk), dimension + 1), dtype=dtype)
            records.view("<i4")[:, 0] = dimension
            records[:, 1:] = chunk
            records.tofile(stream)


def write_fvecs(path, vectors):
    _write_vecs(path, vectors, "<f4")


def write_ivecs(path, vectors):
    _write_vecs(path, vectors, "<i4")
