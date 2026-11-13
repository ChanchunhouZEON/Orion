<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="../orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> Experimental supplier STAG v4

The format is deliberately independent of Orion's native slab alignment. Every
integer in the binary envelope/rows is unsigned little-endian u32. No padding,
compression, offsets or native pointers are stored.

| Field | Encoding |
|---|---|
| Magic | u32 `0x53544147` (bytes `47 41 54 53`, matching existing STAG) |
| Version | u32 `4` |
| Descriptor length | u32, 1..16384 bytes |
| Descriptor | Exactly that many UTF-8 JSON bytes |
| Rows | Exactly `points` rows, ordered by internal node ID |

Each row stores `local_count`, `remote_count`, `extra_count`, then the IDs of
local, remote, extra, in that order. An empty partition has count zero and no IDs.
EOF must follow the last row; trailing bytes and truncated streams are errors.

Example descriptor:

```json
{
  "origin": {"name": "example-supplier", "revision": "r1"},
  "points": 4,
  "dimension": 2,
  "element": "f32",
  "metric": "squared-l2",
  "entry": 2,
  "max_degree": 2,
  "max_extra": 1,
  "base_sha256": null
}
```

`origin.name` is nonblank printable UTF-8, at most 256 bytes; optional revision is
at most 256 bytes without control characters. Neither becomes a path. Fields are
strict: unknown metadata fields are rejected. `element` is `f32` or `u8`;
`metric` is `squared-l2`, `inner-product` or `cosine`; all three support both
f32 and u8. Cosine declares normalized graph geometry: f32 uses unit-normalized
construction coordinates, while native u8 retains raw bytes and uses separate
squared norms for cosine distances. Base identity hashes the original input
before normalization or padding. `points` excludes 0 and the u32 sentinel; dimension/degree are positive.
Entry and every neighbor must be below points. Per row:

- `local_count + remote_count <= max_degree`;
- `extra_count <= max_extra`;
- no self edges or duplicate IDs, including across partitions.

Capacities can exceed actual row degree. Orion validates checked slab-size
arithmetic and the runtime applies the configured memory plan before allocating
the graph. Structural checks cannot certify a useful graph or accurate partitioning.

`base_sha256`, when provided, is 64 lowercase hex digits. For file input, hash the
**entire original input file**, including its original format header(s), before
normalization or padding. File-format conversion changes this digest. For array
inputs, Orion's canonical array identity is different; use null unless supplying
the runtime's matching array identity. Null means the caller asserts row order
and association; shape validation is not base-content verification. The digest
is corruption/mismatch detection, not proof of a supplier's authenticity.

Minimal producer (Python standard library):

```python
import json, struct

def encode(file, spec, rows):
    metadata = json.dumps(spec, separators=(",", ":")).encode("utf-8")
    file.write(struct.pack("<III", 0x53544147, 4, len(metadata)))
    file.write(metadata)
    for local, remote, extra in rows:
        file.write(struct.pack("<III", len(local), len(remote), len(extra)))
        ids = [*local, *remote, *extra]
        file.write(struct.pack(f"<{len(ids)}I", *ids))
```

The producer must validate its data and buffer file writes. Orion validates the
artifact again on import. See `orion-py/tests/test_supplier.py` for a complete
producer fixture with SHA-256, f32/u8/cosine and CLI/Python parity tests.

For STAG v3, the six-u32 header is magic, 3, points, max_degree, max_extra, entry;
rows have the same layout. The explicit legacy adapter requires caller-supplied
missing metadata and checks all four shape/entry values against the header. It
never invents source identity or silently substitutes a medoid.
