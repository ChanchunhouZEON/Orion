# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Validate wheels/sdist in fresh venvs outside the checkout. Never publishes."""
import argparse
import email
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

from common import audit_names, checksums, clean_env, extract, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--numpy', default='numpy>=1.24,<3')
parser.add_argument('--python', default=sys.executable)
parser.add_argument('--install-only', action='store_true', help='Validate wheel on another Python/NumPy pair')
args = parser.parse_args()
interpreter = shutil.which(args.python)
if interpreter is None:
    parser.error(f'Python interpreter not found: {args.python}')
# The test runs outside the checkout; resolve a relative executable beforehand.
interpreter = str(Path(interpreter).absolute())
artifacts = args.artifacts.resolve()
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
wheels = list(artifacts.glob('*.whl'))
sources = list(artifacts.glob('*.tar.gz'))
assert len(wheels) == 1, wheels
if not args.install_only:
    assert len(sources) == 1, sources
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
    audit_names(names)
    metadata_path = next(n for n in names if n.endswith('.dist-info/METADATA'))
    metadata = email.message_from_bytes(archive.read(metadata_path))
    assert metadata['Name'].replace('_', '-') == 'orion-ann' and metadata['Version'] == '0.1.0'
    for license in ['LICENSE', 'ParlayANN-LICENSE', 'Parlaylib-LICENSE']:
        assert any(n.endswith('/' + license) for n in names), license
    assert 'abi3' in wheel.name, wheel.name
    if sys.platform.startswith('linux'):
        assert 'manylinux' in wheel.name, 'candidate wheel must pass the manylinux audit'

env = clean_env()
with tempfile.TemporaryDirectory(prefix='orion-python-validation-') as temporary:
    scratch = Path(temporary)
    smoke = scratch / 'wheel_smoke.py'
    shutil.copy2(Path(__file__).with_name('wheel_smoke.py'), smoke)

    def install_and_test(candidate, label):
        venv = scratch / label
        run([interpreter, '-m', 'venv', venv], scratch, env)
        python = venv / 'bin/python'
        run([python, '-m', 'pip', 'install', args.numpy], scratch, env)
        run([python, '-m', 'pip', 'install', '--no-deps', candidate], scratch, env)
        run([python, '-m', 'pip', 'check'], scratch, env)
        result = run([python, '-I', smoke], scratch, env, capture=True)
        print(result, flush=True)
        return json.loads(result.strip().splitlines()[-1])

    report = {'wheel': install_and_test(wheel, 'wheel-venv')}
    if not args.install_only:
        source_root = scratch / 'source'
        source_root.mkdir()
        names = extract(sources[0], source_root)
        audit_names(names)
        for suffix in ['/cpp/bridge.cpp', '/vendor/algorithms/vamana/index.h',
                       '/vendor/PARLAYANN-LICENSE', '/vendor/PARLAYLIB-LICENSE']:
            assert any(n.endswith(suffix) for n in names), f'sdist missing {suffix}'
        # A fresh build target and pip's isolated PEP 517 environment ensure the
        # source archive can compile without the checkout or an existing binary.
        source = source_root / names[0].split('/')[0]
        run(['cargo', 'metadata', '--locked', '--format-version', '1',
             '--manifest-path', source / 'orion-py/Cargo.toml'], scratch, env, capture=True)
        builder = scratch / 'builder'
        run([interpreter, '-m', 'venv', builder], scratch, env)
        env['CARGO_TARGET_DIR'] = str(scratch / 'sdist-target')
        rebuilt = output / 'from-sdist'
        rebuilt.mkdir(exist_ok=True)
        run([builder / 'bin/python', '-m', 'pip', 'wheel', '--no-deps',
             '--config-settings=build-args=--locked', '--wheel-dir', rebuilt, source], scratch, env)
        rebuilt_wheels = list(rebuilt.glob('*.whl'))
        assert len(rebuilt_wheels) == 1
        report['sdist'] = install_and_test(rebuilt_wheels[0], 'sdist-venv')
    report['status'] = 'passed'
    report['numpy_requirement'] = args.numpy
    report['candidate_hashes'] = checksums(artifacts)
    if not args.install_only:
        report['rebuilt_hashes'] = checksums(output / 'from-sdist')
        report['all_artifact_hashes'] = checksums(output)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
