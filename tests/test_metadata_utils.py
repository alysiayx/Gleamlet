from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from gleamlet.preprocessing.modules._metadata_utils import (
    extract_col_metadata,
    update_col_metadata_entry,
)
from gleamlet.utils.constants import ColumnMetadata


class ExtractColumnMetadataTests(unittest.TestCase):
    def test_cleaned_columns_require_existing_source_mapping(self) -> None:
        metadata = update_col_metadata_entry(
            data_category="attendance",
            std_colnames=["student_id"],
            std_filename="attendance.xlsx",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "columns.xlsx"
            with self.assertRaisesRegex(AssertionError, "Source Column column is completely empty"):
                extract_col_metadata(metadata, str(path))
            self.assertFalse(path.exists())

    def test_cleaned_columns_preserve_existing_source_names(self) -> None:
        original = update_col_metadata_entry(
            data_category="attendance",
            std_colnames=["student_id", "value"],
            raw_colnames=["Student ID", "Value"],
            raw_filename="source.xlsx",
        )
        cleaned = update_col_metadata_entry(
            data_category="attendance", std_colnames=["student_id"]
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "columns.xlsx"
            extract_col_metadata(original, path)
            extract_col_metadata(cleaned, path)
            result = pd.read_excel(path, sheet_name="attendance")
            self.assertEqual(result[ColumnMetadata.STD_NAME].tolist(), ["student_id"])
            self.assertEqual(result[ColumnMetadata.SRC_NAME].tolist(), ["Student ID"])
            self.assertEqual(result[ColumnMetadata.SRC_FILE].tolist(), ["source.xlsx"])

    def test_invalid_metadata_fails_before_creating_or_changing_workbook(self) -> None:
        valid = update_col_metadata_entry(
            data_category="attendance",
            std_colnames=["student_id"],
            raw_colnames=["Student ID"],
        )
        invalid = update_col_metadata_entry(
            data_category="other",
            std_colnames=["first", "second"],
            raw_colnames=["duplicate", "duplicate"],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "columns.xlsx"
            with self.assertRaisesRegex(ValueError, "Duplicate values"):
                extract_col_metadata(invalid, path)
            self.assertFalse(path.exists())
            extract_col_metadata(valid, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Duplicate values"):
                extract_col_metadata({**valid, **invalid}, path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
