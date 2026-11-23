# Copyright (c) Chanchunhou. All rights reserved.
# Licensed under the MIT License.

"""Keep archive acceptance aligned with workspace members, including private SSD."""
import unittest
import tomllib
from pathlib import Path
from common import PACKAGES

class PackageManifestTests(unittest.TestCase):
    def test_workspace_archive_inventory(self):
        root = Path(__file__).resolve().parents[3]
        workspace = tomllib.loads((root / 'Cargo.toml').read_text())['workspace']
        excluded = {'adsampling', 'orion-ann-python', 'orion-ann-wasm'}
        names = set()
        for member in workspace['members']:
            package = tomllib.loads((root / member / 'Cargo.toml').read_text())['package']
            if package['name'] not in excluded:
                names.add(package['name'])
        self.assertEqual(PACKAGES, names)
        ssd = tomllib.loads((root / 'orion-ssd/Cargo.toml').read_text())['package']
        self.assertIn(ssd['name'], PACKAGES)
        self.assertIs(ssd['publish'], False)
