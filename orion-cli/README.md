<!--
Copyright (c) Chanchunhou. All rights reserved.
Licensed under the MIT License.
-->

# <img src="https://raw.githubusercontent.com/ChanchunhouZEON/Orion/main/orion-core/assets/orion-astraeus-heading.svg" alt="" width="72" align="middle" /> orion-ann-cli

Command-line interface for Orion ANN.

Part of [Orion ANN](https://github.com/ChanchunhouZEON/Orion). Requires Rust 1.92 or newer. Licensed under MIT.

Install with `cargo install orion-ann-cli --version 0.1.0`. The executable remains
`orion`; run `orion --help`. `--error-format json` emits the same structured error
report used by Rust and Python. The benchmark sweep is not part of this package.

YAML index configuration uses the same `IndexOptions` schema as the Rust API:

```bash
orion --config options.yaml --base /data/base.u8bin --dimension 128 \
  --max-points 1000000 --preflight --preflight-format table
```

Explicit CLI flags override individual YAML fields; omitted fields use defaults.
Per-object placements merge by object name. Input paths and query controls remain
CLI arguments. `k`, search list size, scores, filters and execution controls are
per-request settings; `--config` does not define a complete search job. YAML uses
Rust field names and nesting (`storage`, `graph.degree`, `storage_options.objects`),
not CLI flag names. Relative paths use the process working directory. See the
[runtime configuration guide](../orion-runtime/src/guide/configuration.md) and
[placement example](../orion-runtime/examples/placement.yaml).
