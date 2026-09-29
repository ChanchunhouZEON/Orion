#!/usr/bin/env bash
# Shared wrapper contract: paths are relative to the repository, never the caller.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
source "$SCRIPT_DIR/_env.sh"

binary_path() {
  local target_dir="${BUILD_DIR:-${CARGO_TARGET_DIR:-$ROOT/target}}"
  [[ "$target_dir" = /* ]] || target_dir="$ROOT/$target_dir"
  printf '%s/release/%s\n' "$target_dir" "$1"
}

build_binary() {
  cargo build --release --bin "$1" --target-dir "${BUILD_DIR:-${CARGO_TARGET_DIR:-$ROOT/target}}"
}

# Resolve through Rust so config overrides, preset paths and metrics have one owner.
# JSON is passed as data; never eval/source values from a configuration file.
paths_for() {
  setup_python_env json >&2
  local config_bin="${ORION_CONFIG_BIN:-$(binary_path orion)}"
  if [[ ! -x "$config_bin" ]]; then
    if [[ -n "${ORION_CONFIG_BIN:-}" ]]; then
      echo "ORION_CONFIG_BIN is not executable: $config_bin" >&2
      return 2
    fi
    build_binary orion >&2
  fi
  local resolved paths
  resolved=$("$config_bin" "$1" --print-config) || return
  paths=$(printf '%s' "$resolved" | "$PY" -c '
import json, sys
config = json.load(sys.stdin)
for key in ("base", "query", "groundtruth"):
    value = config.get(key)
    if not value or "\n" in value:
        sys.exit("Benchmark requires a single file path for " + key)
    print(value)
') || return
  { IFS= read -r BASE; IFS= read -r QRY; IFS= read -r GT; } <<< "$paths"
}
