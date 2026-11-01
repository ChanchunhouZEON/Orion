"""Regression coverage for the Python source-workspace lockfile guard."""
import unittest

from python_source import validate_pruned_lock

BASE = '''version = 4
[[package]]
name = "retained"
version = "1.0.0"
source = "registry+https://example.invalid/index"
checksum = "original"
dependencies = ["shared"]
[[package]]
name = "shared"
version = "2.0.0"
'''
EXTRA = '''
[[package]]
name = "wasm-only"
version = "3.0.0"
'''


class PrunedLockTests(unittest.TestCase):
    def test_unused_workspace_dependency_can_be_removed(self):
        self.assertEqual(validate_pruned_lock(BASE + EXTRA, BASE), ['wasm-only@3.0.0'])
        self.assertEqual(validate_pruned_lock(BASE, BASE), [])

    def test_dependency_addition_or_upgrade_is_rejected(self):
        for changed in [BASE + EXTRA, BASE.replace('2.0.0', '2.1.0')]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_pruned_lock(BASE, changed)

    def test_retained_package_metadata_cannot_drift(self):
        for original, replacement in [('original', 'modified'),
                                       ('example.invalid', 'different.invalid'),
                                       ('dependencies = ["shared"]', 'dependencies = []')]:
            with self.subTest(field=original), self.assertRaises(ValueError):
                validate_pruned_lock(BASE, BASE.replace(original, replacement))
        with self.assertRaises(ValueError):
            validate_pruned_lock(BASE, BASE.replace('version = 4', 'version = 3'))


if __name__ == '__main__':
    unittest.main()
