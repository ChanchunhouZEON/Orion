#!/usr/bin/env bash
# CI smoke test — runs inside the x86_64 Linux container (see
# Dockerfile.ci). Validates that the workspace **cross-compiles
# correctly to x86_64-unknown-linux-gnu** end-to-end:
#
#   1. `vector/` (NEON gated out, AVX2/scalar fallbacks active) compiles.
#   2. `staged_diskann/` (RaBitQ, JL, visited_set quantized codecs) compiles.
#   3. The `benchmark` binary (the SIFT entry point) links cleanly.
#
# Each phase is timed so we know which step blew the budget under
# QEMU TCG emulation. Each phase is COMPILE-ONLY — we deliberately
# skip running the test binaries, because QEMU TCG's `qemu64` CPU
# model lacks AVX/AVX2/FMA. Building with `-C target-cpu=x86-64`
# still leaves rustc-emitted prelude code with newer instructions,
# so even baseline-compiled binaries SIGILL on first run under
# QEMU. Running the AVX-class kernels under QEMU_CPU=max emulates
# successfully but at ~1000× slowdown — also unusable.
#
# Real x86_64 runtime validation:
#   * `.github/workflows/avx512.yml` runs cargo test on GH-hosted
#     Skylake runners (real x86, with AVX-512F / BW / DQ / VL).
#   * The numerical-parity job there compares each SIMD impl
#     against the scalar reference — catches any AVX-512 bug
#     that the compile-only path here cannot.
#
# So the Colima role is: *did this cross-compile change break the
# x86 build?* — a fast (~3 min warm-cache) sanity check before
# pushing to GH Actions for actual binary execution.

set -euo pipefail

# Belt-and-suspenders for the QEMU core-leak issue (see Dockerfile.ci
# notes block). Even if the caller forgot `--ulimit core=0`, the
# shell here drops its core-dump limit before any binary runs, so a
# QEMU SIGILL inside the container cannot drop a `core` file back
# onto the bind-mounted host workspace.
ulimit -c 0 2>/dev/null || true

# `time` builtin output goes to stderr; consolidate so we keep
# a clean log.
exec 2>&1

# Tunables for the smoke phase.
SMOKE_POINTS="${SMOKE_POINTS:-10000}"        # base subsample
SMOKE_QUERIES="${SMOKE_QUERIES:-100}"        # query subsample
SMOKE_LS="${SMOKE_LS:-32,64,128}"            # search list sweep

# Friendly banner.
banner() {
    printf '\n──── %s ────\n' "$*"
}

cpu_info() {
    banner "CPU / kernel info"
    grep -m1 "model name" /proc/cpuinfo || true
    head -1 /proc/cpuinfo
    grep -m1 "^flags" /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|sse|fma)' | sort -u || true
    uname -srm
}

phase_vector_check() {
    banner "Phase 1: cargo check -p vector (x86_64 fallback paths)"
    time cargo check -p vector --release
}

phase_vector_check_no_default_runtime_detect() {
    # The x86_64 distance.rs uses `is_x86_feature_detected!` to
    # pick AVX2 vs scalar at runtime — verify the scalar arm
    # itself type-checks with feature-gated `--no-default-features`.
    banner "Phase 2: cargo check -p vector --no-default-features"
    time cargo check -p vector --release --no-default-features
}

phase_staged_check() {
    banner "Phase 3: cargo check -p staged_diskann (quantized codecs)"
    time cargo check -p staged_diskann --release
}

phase_diskann_check() {
    # Confirms our `cblas` feature gate works — the default build
    # path should compile without OpenBLAS / cblas crates.
    banner "Phase 4: cargo check -p diskann (no-cblas default)"
    time cargo check -p diskann --release
}

phase_build_benchmark() {
    banner "Phase 5: cargo build -p benchmark --release --bin benchmark"
    time cargo build -p benchmark --release --bin benchmark
    # Verify the produced ELF is actually x86_64.
    file ./target-linux/release/benchmark || true
}

phase_sift_smoke_note() {
    banner "Phase 6: SIFT smoke run — SKIPPED on QEMU"
    printf 'QEMU TCG cannot run the produced binary reliably:\n'
    printf '  * qemu64 default CPU lacks AVX2/FMA → SIGILL on rustc-emitted prelude\n'
    printf '  * QEMU_CPU=max emulates AVX-512 but at ~1000x slowdown → unusable\n'
    printf '\n'
    printf 'Real x86_64 runtime validation runs on GH Actions Skylake hosts via\n'
    printf '.github/workflows/avx512.yml (cargo test + numerical-parity job).\n'
    printf '\n'
    printf 'The cross-compile success of phases 1-5 means: any cfg-gate bug or\n'
    printf 'missing trait impl on the x86_64 path would have shown up here.\n'
}

cpu_info
phase_vector_check
phase_vector_check_no_default_runtime_detect
phase_staged_check
phase_diskann_check
phase_build_benchmark
phase_sift_smoke_note

banner "SMOKE TEST COMPLETE (compile-only on QEMU)"
