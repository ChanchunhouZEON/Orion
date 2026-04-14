#!/usr/bin/env python3
"""Convert ann-benchmarks HDF5 datasets to fvecs/ivecs format.

Usage: python3 convert_hdf5.py input.hdf5 output_dir [--normalize]

--normalize: L2-normalize all vectors (makes L2 distance equivalent to angular).
"""
import h5py
import numpy as np
import sys
import os


def write_fvecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = np.array([len(vec)], dtype=np.int32)
            dim.tofile(f)
            vec.astype(np.float32).tofile(f)


def write_ivecs(filename, vecs):
    with open(filename, "wb") as f:
        for vec in vecs:
            dim = np.array([len(vec)], dtype=np.int32)
            dim.tofile(f)
            vec.astype(np.int32).tofile(f)


SUPPORTED_DIMS = [25, 32, 100, 104, 128, 256, 784, 960]


def pad_to_supported_dim(vecs):
    """Zero-pad vectors to the nearest supported dimension."""
    d = vecs.shape[1]
    target = min((s for s in SUPPORTED_DIMS if s >= d), default=None)
    if target is None or target == d:
        return vecs
    print(f"Padding dimension {d} -> {target}")
    pad = np.zeros((vecs.shape[0], target - d), dtype=vecs.dtype)
    return np.hstack([vecs, pad])


def convert(hdf5_path, output_dir, normalize=False):
    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(hdf5_path))[0]

    with h5py.File(hdf5_path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)

        if normalize:
            norms = np.linalg.norm(train, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            train = train / norms
            norms = np.linalg.norm(test, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            test = test / norms
            print(f"Normalized to unit vectors")

        train = pad_to_supported_dim(train)
        test = pad_to_supported_dim(test)

        print(f"Base: {train.shape}, Queries: {test.shape}, GT: {neighbors.shape}")

        write_fvecs(os.path.join(output_dir, f"{name}_base.fvecs"), train)
        write_fvecs(os.path.join(output_dir, f"{name}_query.fvecs"), test)
        write_ivecs(os.path.join(output_dir, f"{name}_groundtruth.ivecs"), neighbors)

    print(f"Written to {output_dir}/")


if __name__ == "__main__":
    normalize = "--normalize" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    convert(args[0], args[1] if len(args) > 1 else ".", normalize)
