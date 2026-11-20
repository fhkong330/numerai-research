import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import paths


class VersionAwarePathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temp_dir.name)
        self.data_root = self.project_root / "data"
        self.output_root = self.project_root / "outputs"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def path_roots(self):
        return (
            patch.object(paths, "DATA_ROOT", self.data_root),
            patch.object(paths, "OUTPUT_ROOT", self.output_root),
        )

    def test_v5_3_uses_flat_paths_before_relocation(self) -> None:
        data_patch, output_patch = self.path_roots()
        with data_patch, output_patch:
            self.assertEqual(paths.resolve_data_dir("v5.3"), self.data_root)
            self.assertEqual(paths.resolve_output_dir("v5.3"), self.output_root)

    def test_v5_3_uses_versioned_paths_after_relocation(self) -> None:
        (self.data_root / "v5.3").mkdir(parents=True)
        (self.output_root / "v5.3").mkdir(parents=True)

        data_patch, output_patch = self.path_roots()
        with data_patch, output_patch:
            self.assertEqual(
                paths.resolve_data_dir("v5.3"),
                self.data_root / "v5.3",
            )
            self.assertEqual(
                paths.resolve_output_dir("v5.3"),
                self.output_root / "v5.3",
            )

    def test_v6_0_always_uses_isolated_versioned_paths(self) -> None:
        data_patch, output_patch = self.path_roots()
        with data_patch, output_patch:
            self.assertEqual(
                paths.resolve_data_dir("v6.0"),
                self.data_root / "v6.0",
            )
            self.assertEqual(
                paths.resolve_output_dir("v6.0"),
                self.output_root / "v6.0",
            )
            self.assertFalse((self.data_root / "v6.0").exists())
            self.assertFalse((self.output_root / "v6.0").exists())

    def test_versions_resolve_to_different_paths(self) -> None:
        data_patch, output_patch = self.path_roots()
        with data_patch, output_patch:
            self.assertNotEqual(
                paths.resolve_data_dir("v5.3"),
                paths.resolve_data_dir("v6.0"),
            )
            self.assertNotEqual(
                paths.resolve_output_dir("v5.3"),
                paths.resolve_output_dir("v6.0"),
            )

    def test_unsupported_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported dataset version"):
            paths.resolve_data_dir("v7.0")


if __name__ == "__main__":
    unittest.main()
