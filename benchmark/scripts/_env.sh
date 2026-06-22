#!/usr/bin/env bash
# Shared Python-environment setup. Source this from any wrapper:
#
#   source "$(dirname "$0")/_env.sh"
#   setup_python_env           # default: requires matplotlib + numpy
#   setup_python_env numpy     # custom probe (e.g. data-prep scripts)
#
# Resolves $PY to an interpreter that can `import` the requested
# modules. Honors user-provided $PY first; otherwise walks a list of
# common locations (PATH `python3`, Homebrew, conda). On failure
# prints a single, actionable hint pointing at requirements.txt.
#
# Idempotent — safe to call multiple times in the same script.

setup_python_env() {
  # Probe modules: positional args, or "matplotlib numpy" by default.
  local probe_modules=("$@")
  if [[ ${#probe_modules[@]} -eq 0 ]]; then
    probe_modules=(matplotlib numpy)
  fi
  local import_stmt
  import_stmt=$(printf 'import %s; ' "${probe_modules[@]}")

  # Candidate interpreters. User-provided $PY wins; otherwise scan a
  # small ordered list. PATH `python3` first so an active virtualenv
  # is honored without ceremony.
  local candidates
  if [[ -n "${PY:-}" ]]; then
    candidates=("$PY")
  else
    candidates=(
      python3
      /opt/homebrew/bin/python3
      /usr/local/bin/python3
      /opt/anaconda3/envs/ray/bin/python
      /usr/bin/python3
    )
  fi

  for cand in "${candidates[@]}"; do
    if command -v "$cand" &>/dev/null \
        && "$cand" -c "$import_stmt" &>/dev/null; then
      PY="$cand"
      export PY
      echo "[env] PY=$PY  (validated: ${probe_modules[*]})"
      return 0
    fi
  done

  cat >&2 <<EOF
[env] No Python interpreter with required modules importable.
      Probed:   ${candidates[*]}
      Required: ${probe_modules[*]}

      Install the visualization deps with:
          python3 -m pip install -r requirements.txt
      Then override the interpreter via:
          PY=/path/to/python bash $0
EOF
  return 1
}
