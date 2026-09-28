#!/usr/bin/env python3
"""Bounded-memory conversion to ParlayANN binary vectors and ground truth.

Byte coordinates stay bytes; no whole-file read or whole-dataset f32 expansion.
The legacy --base-fvecs/--query-fvecs flags remain aliases for existing recipes.
"""
import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import struct
import tempfile

import numpy as np

FORMATS = {"fvecs": (4, True), "bvecs": (1, True),
           "fbin": (4, False), "u8bin": (1, False), "ivecs": (4, True)}
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


@contextmanager
def atomic_output(path):
    """Keep the previous output intact on validation, write, or flush failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            yield output
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inspect_vectors(path, vector_format):
    coordinate_bytes, per_record_header = FORMATS[vector_format]
    file_bytes = Path(path).stat().st_size
    with open(path, "rb") as source:
        header = source.read(4 if per_record_header else 8)
    if len(header) != (4 if per_record_header else 8):
        raise ValueError("truncated vector header")

    if per_record_header:
        dimension, = struct.unpack("<I", header)
        stride = 4 + dimension * coordinate_bytes
        count, remainder = divmod(file_bytes, stride)
        if remainder:
            raise ValueError("truncated vector records")
    else:
        count, dimension = struct.unpack("<II", header)
        if file_bytes != 8 + count * dimension * coordinate_bytes:
            raise ValueError("binary header/file length mismatch")

    if not dimension or not count or count >= 2**32 - 1:
        raise ValueError("empty vectors or count outside supported u32 IDs")
    return count, dimension


def payload_chunks(path, vector_format, count, dimension, chunk_bytes):
    coordinate_bytes, per_record_header = FORMATS[vector_format]
    stride = dimension * coordinate_bytes + (4 if per_record_header else 0)
    rows_per_chunk = max(1, chunk_bytes // stride)
    with open(path, "rb") as source:
        if not per_record_header:
            source.seek(8)
        for first in range(0, count, rows_per_chunk):
            rows = min(rows_per_chunk, count - first)
            block = source.read(rows * stride)
            if len(block) != rows * stride:
                raise ValueError("truncated vector payload")
            if per_record_header:
                records = np.frombuffer(block, dtype=np.uint8).reshape(rows, stride)
                headers = records[:, :4].copy().view("<u4").reshape(-1)
                if not np.all(headers == dimension):
                    raise ValueError("inconsistent per-record dimension")
                # The contiguous copy is bounded by this chunk, never the dataset.
                yield records[:, 4:].tobytes()
            else:
                yield block


def convert_vectors(source, destination, vector_format, limit=0, chunk_bytes=DEFAULT_CHUNK_BYTES):
    if limit < 0 or chunk_bytes <= 0:
        raise ValueError("limit must be nonnegative and chunk bytes positive")
    count, dimension = inspect_vectors(source, vector_format)
    count = min(count, limit) if limit else count
    with atomic_output(destination) as output:
        output.write(struct.pack("<II", count, dimension))
        for payload in payload_chunks(source, vector_format, count, dimension, chunk_bytes):
            output.write(payload)
    return count, dimension


def convert_ground_truth(source, destination, query_count, base_count, chunk_bytes=DEFAULT_CHUNK_BYTES):
    count, neighbors = inspect_vectors(source, "ivecs")
    if count < query_count:
        raise ValueError("ground truth has fewer records than queries")
    with atomic_output(destination) as output:
        output.write(struct.pack("<II", query_count, neighbors))
        for payload in payload_chunks(source, "ivecs", query_count, neighbors, chunk_bytes):
            if np.any(np.frombuffer(payload, dtype="<u4") >= base_count):
                raise ValueError("ground truth IDs exceed loaded base; supply ground truth for this exact prefix")
            output.write(payload)
        # ParlayANN recall consumes IDs; its binary GT layout also requires distances.
        remaining = query_count * neighbors * 4
        zeros = bytes(min(chunk_bytes, remaining))
        while remaining:
            block = zeros[:min(len(zeros), remaining)]
            output.write(block)
            remaining -= len(block)
    return query_count, neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", "--base-fvecs", dest="base", required=True)
    parser.add_argument("--query", "--query-fvecs", dest="query", required=True)
    parser.add_argument("--base-format", choices=[x for x in FORMATS if x != "ivecs"])
    parser.add_argument("--query-format", choices=[x for x in FORMATS if x != "ivecs"])
    parser.add_argument("--gt-ivecs", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-base-points", type=int, default=0)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    args = parser.parse_args()
    base_format = args.base_format or Path(args.base).suffix.lstrip(".")
    query_format = args.query_format or Path(args.query).suffix.lstrip(".")
    if base_format not in FORMATS or query_format not in FORMATS:
        parser.error("unknown vector format; specify --base-format/--query-format")
    if args.chunk_bytes <= 0 or args.max_base_points < 0:
        parser.error("invalid chunk bytes or point limit")
    if FORMATS[base_format][0] != FORMATS[query_format][0]:
        parser.error("base and query must use the same coordinate representation")
    base_count, dimension = inspect_vectors(args.base, base_format)
    query_count, query_dimension = inspect_vectors(args.query, query_format)
    if dimension != query_dimension:
        parser.error("base/query dimension mismatch")
    base_count = min(base_count, args.max_base_points) if args.max_base_points else base_count
    out = Path(args.out_dir)
    suffix = "u8bin" if FORMATS[base_format][0] == 1 else "fbin"

    # Validate small GT before spending hours on a large base conversion.
    convert_ground_truth(args.gt_ivecs, out / "gt.bin", query_count, base_count, args.chunk_bytes)
    convert_vectors(args.query, out / ("query." + suffix), query_format, chunk_bytes=args.chunk_bytes)
    convert_vectors(args.base, out / ("base." + suffix), base_format,
                    args.max_base_points, args.chunk_bytes)
    print(f"Converted {base_count} x {dimension} base and {query_count} queries to {suffix}: {out}")


if __name__ == "__main__":
    main()
