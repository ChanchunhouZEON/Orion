"""Verify all public packaged crates and an external consumer. Never uploads."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import tomllib

from common import PACKAGES, audit_names, checksums, clean_env, extract, run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--archives', type=Path, help='Reuse an already cargo-verified set of archives')
parser.add_argument('--allow-dirty', action='store_true')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[3]
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=True)
env = clean_env()
with tempfile.TemporaryDirectory(prefix='orion-crate-validation-') as temporary:
    scratch = Path(temporary)
    if args.archives:
        package_dir = args.archives.resolve()
    else:
        target = scratch / 'package-target'
        command = ['cargo', 'package', '--locked', '--workspace', '--exclude', 'benchmark',
                   '--exclude', 'adsampling', '--exclude', 'orion-ann-python',
                   '--exclude', 'orion-ann-wasm', '--target-dir', target]
        if args.allow_dirty:
            command.append('--allow-dirty')
        run(command, repo, env)
        package_dir = target / 'package'
    archives = sorted(package_dir.glob('*.crate'))
    assert len(archives) == len(PACKAGES), [p.name for p in archives]
    extracted = scratch / 'extracted'
    extracted.mkdir()
    packages = {}
    for archive in archives:
        names = extract(archive, extracted)
        audit_names(names)
        root = extracted / names[0].split('/')[0]
        manifest = tomllib.loads((root / 'Cargo.toml').read_text())
        package = manifest['package']
        name = package['name']
        assert package['version'] == '0.1.0' and name in PACKAGES, package
        assert (root / 'LICENSE').is_file() and (root / 'README.md').is_file(), name
        packages[name] = root
        shutil.copy2(archive, output / archive.name)
    assert set(packages) == PACKAGES, packages
    assert (packages['orion-ann-parlay'] / 'cpp/bridge.cpp').is_file()
    assert (packages['orion-ann-parlay'] / 'vendor/algorithms/vamana/index.h').is_file()
    assert (packages['orion-ann-runtime'] / 'configs/sweep.yaml').is_file()

    # Patch only the unpublished Orion packages, using extracted .crate payloads.
    # The consumer has no dependency on the checkout, scripts, or its .cargo flags.
    patches = '[patch.crates-io]\n' + ''.join(
        f'{name} = {{ path = {json.dumps(str(path))} }}\n' for name, path in sorted(packages.items()))
    consumer = scratch / 'consumer'
    (consumer / 'src').mkdir(parents=True)
    (consumer / 'Cargo.toml').write_text('''[package]
name = "orion-release-consumer"
version = "0.0.0"
edition = "2021"
[dependencies]
orion-ann-runtime = "=0.1.0"
orion-ann = "=0.1.0"
''' + patches)
    shutil.copy2(Path(__file__).with_name('rust_smoke.rs'), consumer / 'src/main.rs')
    env['CARGO_TARGET_DIR'] = str(scratch / 'consumer-target')
    run(['cargo', 'run', '--manifest-path', consumer / 'Cargo.toml'], consumer, env)
    metadata = json.loads(run(['cargo', 'metadata', '--format-version', '1'], consumer, env, capture=True))
    for package in metadata['packages']:
        if package['name'] in PACKAGES:
            assert Path(package['manifest_path']).is_relative_to(extracted), package
    patch_file = scratch / 'patches.toml'
    patch_file.write_text(patches)
    installed = scratch / 'installed'
    run(['cargo', 'install', '--path', packages['orion-ann-cli'], '--debug',
         '--root', installed, '--config', patch_file], consumer, env)
    run([installed / 'bin/orion', '--help'], scratch, env)
    report = {'status': 'passed', 'packages': sorted(packages),
              'consumer': 'f32/u8 default ParlayANN; calibration/search/snapshot',
              'cli_install': 'passed', 'artifacts': checksums(output)}
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
