# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.
"""Build private ctypes adapters. Never changes either engine's source tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parlay-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    native = Path(__file__).resolve().parent / 'native'
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = '.dylib' if platform.system() == 'Darwin' else '.so'
    parlay = args.output.resolve() / ('liblatency_parlay' + suffix)
    commands = [
        [os.environ.get('CXX', 'c++'), '-std=c++20', '-O3', '-DNDEBUG', '-fPIC', '-shared',
         '-pthread', *shlex.split(os.environ.get('CXXFLAGS', '-march=native')),
         '-I' + str(args.parlay_root.resolve()),
         '-I' + str(args.parlay_root.resolve() / 'parlaylib/include'),
         str(native / 'parlay.cpp'), '-o', str(parlay)],
        ['cargo', '+1.92', 'build', '--release', '--locked', '--manifest-path',
         str(native / 'diskann/Cargo.toml'), '--target-dir', str(args.output.resolve() / 'cargo')],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    diskann = args.output.resolve() / 'cargo/release' / ('libbaseline_latency_diskann' + suffix)
    files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (parlay, diskann)}
    record = dict(commands=commands, files=files, machine=platform.platform(),
                  cxx_version=subprocess.check_output([os.environ.get('CXX', 'c++'), '--version'], text=True),
                  rustflags=os.environ.get('RUSTFLAGS', ''),
                  parlay_commit=subprocess.check_output(
                      ['git', '-C', str(args.parlay_root), 'rev-parse', 'HEAD'], text=True).strip())
    (args.output / 'build.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(files, indent=2))


if __name__ == '__main__':
    main()
