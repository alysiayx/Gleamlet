import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gleamlet.preprocessing.modules import cleaning
from gleamlet.preprocessing.pipeline import DataPreprocessor


class GlobalConstantCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.output = self.root / "cleaned"
        self.source.mkdir()
        self.output.mkdir()
        self.file = self.source / "2024_attendance.xlsx"
        pd.DataFrame({"stud_id": [1, 2], "constant": [7, 7]}).to_excel(
            self.file, index=False
        )
        self.cache = self.root / ".cleaned_global_constants.json"
        self.scan = self.enterContext(patch.object(
            cleaning, "identify_globally_constant_columns", return_value=["constant"]
        ))
        self.enterContext(patch.object(cleaning, "extract_col_metadata"))
        self.enterContext(patch.object(cleaning, "update_col_metadata_entry", return_value={}))

    def clean(self, **kwargs):
        options = dict(
            input_path=self.source,
            output_path=self.output,
            col_metadata_path=self.root / "metadata.xlsx",
            file_naming_format=["cohort_y11_ay", "data_category"],
            rm_constant_cols="global",
            overwrite=True,
        )
        options.update(kwargs)
        cleaning.clean_data(**options)

    def test_disk_cache_reused_with_overwrite_and_same_output(self):
        self.clean()
        expected = pd.read_excel(self.output / self.file.name)
        self.assertNotIn("constant", expected.columns)
        self.assertTrue(self.cache.exists())
        self.clean(rm_sensitive_cols=["absent"])
        self.scan.assert_called_once()
        pd.testing.assert_frame_equal(pd.read_excel(self.output / self.file.name), expected)

    def test_empty_result_is_cached(self):
        self.scan.return_value = []
        self.clean()
        self.clean()
        self.scan.assert_called_once()

    def test_force_refresh(self):
        self.clean()
        self.clean(refresh_constant_cache=True)
        self.assertEqual(self.scan.call_count, 2)

    def test_detection_parameters_invalidate(self):
        self.clean()
        for options in (
            {"constant_consistency": 0.9},
            {"missing_cutoff": 0.9},
            {"consider_missing": True},
        ):
            with self.subTest(options=options):
                self.clean()
                before = self.scan.call_count
                self.clean(**options)
                self.assertEqual(self.scan.call_count, before + 1)

    def test_file_changes_invalidate(self):
        self.clean()
        stat = self.file.stat()
        os.utime(self.file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.clean()
        self.assertEqual(self.scan.call_count, 2)
        # Size changes must invalidate even if the modification time is restored.
        stat = self.file.stat()
        with self.file.open("ab") as stream:
            stream.write(b"extra")
        os.utime(self.file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.clean()
        self.assertEqual(self.scan.call_count, 3)
        other = self.source / "2025_attendance.xlsx"
        other.write_bytes(self.file.read_bytes())
        self.clean()
        self.assertEqual(self.scan.call_count, 4)
        other.rename(self.root / "moved.xlsx")
        self.clean()
        self.assertEqual(self.scan.call_count, 5)
        self.file.rename(self.source / "2026_attendance.xlsx")
        self.clean()
        self.assertEqual(self.scan.call_count, 6)

    def test_input_directory_invalidates(self):
        self.clean()
        moved = self.root / "moved"
        self.source.rename(moved)
        self.clean(input_path=moved)
        self.assertEqual(self.scan.call_count, 2)

    def test_bad_cache_is_rebuilt(self):
        self.clean()
        valid = json.loads(self.cache.read_text())
        for contents in ("broken JSON", "[]", json.dumps({**valid, "columns": [42]})):
            with self.subTest(contents=contents):
                self.cache.write_text(contents)
                before = self.scan.call_count
                self.clean()
                self.assertEqual(self.scan.call_count, before + 1)

    def test_non_global_modes_do_not_scan_or_cache(self):
        self.clean(rm_constant_cols="local")
        self.clean(rm_constant_cols=False)
        self.scan.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_cache_write_failure_does_not_stop_cleaning(self):
        with patch.object(Path, "write_text", side_effect=OSError("read only")):
            self.clean()
        self.assertTrue((self.output / self.file.name).exists())

    def test_public_method_forwards_refresh(self):
        preprocessor = object.__new__(DataPreprocessor)
        preprocessor.src_colstd_dir = self.source
        preprocessor.clean_data_dir = self.output
        preprocessor.col_metadata_path = self.root / "metadata.xlsx"
        preprocessor.file_naming_format = ["cohort_y11_ay", "data_category"]
        preprocessor.overwrite = True
        with patch("gleamlet.preprocessing.pipeline._clean_data") as clean:
            preprocessor.clean_data(refresh_constant_cache=True)
        self.assertTrue(clean.call_args.kwargs["refresh_constant_cache"])


if __name__ == "__main__":
    unittest.main()
