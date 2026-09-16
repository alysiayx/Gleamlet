from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from gleamlet.preprocessing.modules._metadata_utils import (
    extract_col_metadata,
    update_col_metadata_entry,
)
from gleamlet.utils.constants import ColumnMetadata


class ExtractColumnMetadataTests(unittest.TestCase):
    def test_cleaned_columns_without_source_mapping_use_separate_workbook(self) -> None:
        metadata = update_col_metadata_entry(
            data_category="attendance",
            std_colnames=["student_id"],
            std_filename="attendance.xlsx",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "columns.xlsx"
            with self.assertLogs("data_processor", level="WARNING") as logs:
                extract_col_metadata(metadata, str(path))
            self.assertFalse(path.exists())
            incomplete = path.with_name("columns_draft.xlsx")
            result = pd.read_excel(incomplete, sheet_name="attendance")
            self.assertEqual(result[ColumnMetadata.STD_NAME].tolist(), ["student_id"])
            self.assertTrue(result[ColumnMetadata.SRC_NAME].isna().all())
            self.assertIn(str(incomplete), '\n'.join(logs.output))
            self.assertIn("blank Source Column cells in sheets: attendance", '\n'.join(logs.output))

    def test_notebook_offers_link_after_draft_is_written(self) -> None:
        metadata = update_col_metadata_entry(
            data_category="attendance", std_colnames=["student_id"]
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data_dictionary.xlsx"
            draft = path.with_name("data_dictionary_draft.xlsx")
            display = Mock(side_effect=lambda link: self.assertTrue(draft.exists()))
            file_link = Mock()
            with patch.dict("sys.modules", {
                "IPython": SimpleNamespace(get_ipython=lambda: SimpleNamespace(kernel=object())),
                "IPython.display": SimpleNamespace(FileLink=file_link, display=display),
            }):
                extract_col_metadata(metadata, path)
            display.assert_called_once_with(file_link.return_value)
            linked_path = Path(file_link.call_args.args[0]).resolve()
            self.assertEqual(linked_path, draft.resolve())

    def test_workbook_without_source_column_is_preserved(self) -> None:
        metadata = update_col_metadata_entry(
            data_category="attendance", std_colnames=["student_id"]
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data_dictionary.xlsx"
            pd.DataFrame({ColumnMetadata.STD_NAME: ["student_id"]}).to_excel(
                path, sheet_name="attendance", index=False
            )
            before = path.read_bytes()
            extract_col_metadata(metadata, path)
            self.assertEqual(path.read_bytes(), before)
            result = pd.read_excel(
                path.with_name("data_dictionary_draft.xlsx"),
                sheet_name="attendance",
            )
            self.assertEqual(result[ColumnMetadata.STD_NAME].tolist(), ["student_id"])
            self.assertTrue(result[ColumnMetadata.SRC_NAME].isna().all())

    def test_partial_mapping_is_exported_without_changing_original(self) -> None:
        original = update_col_metadata_entry(
            data_category="attendance",
            std_colnames=["student_id"], raw_colnames=["Student ID"],
        )
        cleaned = update_col_metadata_entry(
            data_category="attendance", std_colnames=["student_id", "new_column"]
        )
        cleaned = update_col_metadata_entry(
            data_category="census", col_meta=cleaned,
            std_colnames=["year"], raw_colnames=["Year"],
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "columns.xlsx"
            extract_col_metadata(original, path)
            before = path.read_bytes()
            extract_col_metadata(cleaned, path)
            self.assertEqual(path.read_bytes(), before)
            result = pd.read_excel(path.with_name("columns_draft.xlsx"), sheet_name=None)
            self.assertEqual(set(result), {"attendance", "census"})
            sources = result["attendance"][ColumnMetadata.SRC_NAME]
            self.assertEqual(sources.iloc[0], "Student ID")
            self.assertTrue(pd.isna(sources.iloc[1]))
            self.assertEqual(result["census"][ColumnMetadata.SRC_NAME].tolist(), ["Year"])
            self.assertEqual(list(Path(folder).glob("*pre_clean*")), [])

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
