import tempfile
import unittest
from pathlib import Path
from unittest.mock import ANY, Mock

from src.download_data import download_files, resolve_dataset_file


class DatasetDownloaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name) / "data"
        self.api = Mock()
        self.api.download_dataset.side_effect = self.write_download

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def write_download(dataset_path: str, destination: str) -> str:
        Path(destination).write_bytes(dataset_path.encode("utf-8"))
        return destination

    def test_v5_3_download_uses_versioned_directory(self) -> None:
        result = download_files(
            "v5.3",
            ["features.json"],
            data_root=self.data_root,
            api=self.api,
        )

        expected = self.data_root / "v5.3" / "features.json"
        self.assertEqual(result["downloaded"], [expected])
        self.assertTrue(expected.exists())
        self.api.download_dataset.assert_called_once_with(
            "v5.3/features.json",
            ANY,
        )

    def test_v6_0_download_uses_versioned_directory(self) -> None:
        result = download_files(
            "v6.0",
            ["v6.0/features.json"],
            data_root=self.data_root,
            api=self.api,
        )

        expected = self.data_root / "v6.0" / "features.json"
        self.assertEqual(result["downloaded"], [expected])
        self.assertTrue(expected.exists())
        self.assertFalse((self.data_root / "v5.3" / "features.json").exists())

    def test_only_explicitly_selected_files_are_downloaded(self) -> None:
        download_files(
            "v6.0",
            ["features.json", "validation_example_preds.parquet"],
            data_root=self.data_root,
            api=self.api,
        )

        requested_paths = [
            call.args[0] for call in self.api.download_dataset.call_args_list
        ]
        self.assertEqual(
            requested_paths,
            [
                "v6.0/features.json",
                "v6.0/validation_example_preds.parquet",
            ],
        )
        self.assertFalse((self.data_root / "v6.0" / "train.parquet").exists())

    def test_existing_file_is_skipped_without_overwrite(self) -> None:
        destination = self.data_root / "v6.0" / "features.json"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b"existing metadata")

        result = download_files(
            "v6.0",
            ["features.json"],
            data_root=self.data_root,
            api=self.api,
        )

        self.assertEqual(result["skipped"], [destination])
        self.assertEqual(destination.read_bytes(), b"existing metadata")
        self.api.download_dataset.assert_not_called()

    def test_unsupported_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported dataset version"):
            download_files(
                "v7.0",
                ["features.json"],
                data_root=self.data_root,
                api=self.api,
            )

        self.assertFalse(self.data_root.exists())
        self.api.download_dataset.assert_not_called()

    def test_unavailable_dataset_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unavailable dataset file"):
            download_files(
                "v6.0",
                ["unknown.parquet"],
                data_root=self.data_root,
                api=self.api,
            )

        self.assertFalse(self.data_root.exists())
        self.api.download_dataset.assert_not_called()

    def test_cross_version_catalog_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not belong to v6.0"):
            resolve_dataset_file("v6.0", "v5.3/features.json")


if __name__ == "__main__":
    unittest.main()
