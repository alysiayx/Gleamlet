from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from gleamlet.preprocessing.modules import naming
from gleamlet.utils.constants import FileMetadata


class StandardiseNamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.input_path = self.root / "source"
        self.output_path = self.root / "standardised"
        self.input_path.mkdir()
        self.output_path.mkdir()

        self.source_path = self.input_path / "source.xlsx"
        pd.DataFrame({"value": [1, 2]}).to_excel(
            self.source_path,
            sheet_name="Data",
            index=False,
        )
        self.file_metadata = pd.DataFrame(
            {
                FileMetadata.CATEGORY: ["attendance"],
                FileMetadata.FILE_NAME: [self.source_path.name],
                FileMetadata.SHEET_NAME: ["Data"],
                FileMetadata.COHORT_Y11_AY: ["2022-2023"],
                FileMetadata.YEAR_GRP: [11],
            }
        )
        self.destination = self.output_path / "2022-2023_attendance.xlsx"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _standardise(self) -> None:
        naming.standardise_fnames_colnames(
            input_path=self.input_path,
            output_path=self.output_path,
            file_metadata=self.file_metadata,
            file_metadata_path=self.root / "source_inventory.xlsx",
            col_metadata_path=self.root / "data_dictionary.xlsx",
            file_naming_format=["cohort_y11_ay", "data_category"],
            standardise_rules=[],
            valid_cat=["attendance"],
            add_prefix=False,
            overwrite=True,
        )

    def test_failed_excel_write_preserves_existing_output(self) -> None:
        original_contents = b"previous-valid-output"
        self.destination.write_bytes(original_contents)
        original_to_excel = pd.DataFrame.to_excel

        def write_then_fail(frame, *args, **kwargs):
            original_to_excel(frame, *args, **kwargs)
            raise RuntimeError("write failed")

        with (
            patch.object(naming, "extract_col_metadata"),
            patch.object(naming, "compare_and_backup"),
            patch.object(pd.DataFrame, "to_excel", new=write_then_fail),
        ):
            with self.assertRaisesRegex(RuntimeError, "write failed"):
                self._standardise()

        self.assertEqual(self.destination.read_bytes(), original_contents)
        self.assertEqual(list(self.output_path.glob(".*.tmp.xlsx")), [])

    def test_successful_excel_write_replaces_existing_output(self) -> None:
        self.destination.write_bytes(b"previous-output")

        with (
            patch.object(naming, "extract_col_metadata"),
            patch.object(naming, "compare_and_backup"),
        ):
            self._standardise()

        pd.testing.assert_frame_equal(
            pd.read_excel(self.destination),
            pd.DataFrame({"value": [1, 2]}),
        )
        self.assertEqual(list(self.output_path.glob(".*.tmp.xlsx")), [])


if __name__ == "__main__":
    unittest.main()
