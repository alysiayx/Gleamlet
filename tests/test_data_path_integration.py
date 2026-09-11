from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from gleamlet.config import GleamletConfig
from gleamlet.features import FeatureEngineer
from gleamlet.preprocessing import DataPreprocessor
from gleamlet.preprocessing.modules._merging_utils import merge_one_item_group
from gleamlet.utils.constants import FileMetadata
from gleamlet.utils.misc import make_output_dir


class DataPathIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name).resolve()
        self.config = GleamletConfig.load(
            project_root=self.project_root,
            user_config_path=self.project_root / ".gleamlet/config.yaml",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_feature_engineer_uses_registry_or_explicit_input(self) -> None:
        expected_input = (
            self.project_root
            / "data/02_processed/2_merged/longitudinal.parquet"
        )
        expected_output = self.project_root / "data/longitudinal.parquet"
        engineer = FeatureEngineer(settings=self.config)
        self.assertEqual(engineer.input_data_path, expected_input)
        self.assertEqual(
            engineer.get_path("output").Path.iloc[0],
            expected_output,
        )

        explicit = self.project_root / "alternate/input.parquet"
        engineer.set_input_data_path(explicit)
        self.assertEqual(engineer.input_data_path, explicit)

        explicit_engineer = FeatureEngineer(
            settings=self.config,
            input_data_path=explicit,
            output_filename="output.parquet",
        )
        self.assertEqual(explicit_engineer.input_data_path, explicit)
        self.assertEqual(
            explicit_engineer.output_data_path,
            self.project_root / "data/output.parquet",
        )

    def test_pipeline_stage_directories_come_from_config(self) -> None:
        self.config.update(paths={
            "cleaned_dir": "stages/cleaned",
            "merged_dir": "stages/merged",
            "linked_dir": "stages/linked",
            "derived_dir": "stages/derived",
            "aggregated_dir": "stages/aggregated",
        })

        curator = DataPreprocessor(settings=self.config)
        engineer = FeatureEngineer(settings=self.config)

        self.assertEqual(curator.clean_data_dir, self.project_root / "stages/cleaned")
        self.assertEqual(curator.merge_data_dir, self.project_root / "stages/merged")
        self.assertEqual(engineer.link_data_dir, self.project_root / "stages/linked")
        self.assertEqual(engineer.derive_data_dir, self.project_root / "stages/derived")
        self.assertEqual(
            engineer.agg_data_dir,
            self.project_root / "stages/aggregated",
        )

    def test_make_output_dir_accepts_directory_or_file_paths(self) -> None:
        output_directory = self.project_root / "outputs/reports"
        output_file = self.project_root / "outputs/tables/metrics.parquet"
        logger = Mock()

        self.assertEqual(
            make_output_dir(output_directory, logger=logger),
            output_directory,
        )
        self.assertEqual(
            make_output_dir(output_file, logger=logger),
            output_file.parent,
        )
        self.assertTrue(output_directory.is_dir())
        self.assertTrue(output_file.parent.is_dir())
        self.assertFalse(output_file.exists())
        self.assertEqual(logger.info.call_count, 2)

        make_output_dir(output_file, logger=logger)
        self.assertEqual(logger.info.call_count, 2)

    def test_merge_available_years_writes_and_persists_inferred_path(self) -> None:
        curator = DataPreprocessor(settings=self.config)
        metadata = pd.DataFrame(
            {
                FileMetadata.COHORT_Y11_AY: [
                    "2018 - 2019",
                    "2025 - 2026",
                    "2026 Mar",
                ]
            }
        )
        captured = {}

        def fake_merge(**kwargs):
            captured.update(kwargs)

        with (
            patch("gleamlet.preprocessing.pipeline.load_dataframe", return_value=metadata),
            patch("gleamlet.preprocessing.pipeline._merge_data", side_effect=fake_merge),
        ):
            curator.merge_data(
                use_file_metadata="metadata.xlsx",
                data_schema={},
                group_by=False,
                name_by_year_range=True,
            )

        expected = self.project_root / "data/02_processed/2_merged/2018-2026.parquet"
        self.assertEqual(captured["output_filename"], expected.name)
        self.assertEqual(
            curator.settings.get_path("prepared"),
            expected,
        )
        reloaded = GleamletConfig.load(
            project_root=self.project_root,
            user_config_path=self.config.user_config_path,
        )
        self.assertEqual(reloaded.get_path("prepared"), expected)
        self.assertEqual(
            reloaded.get_path("model_input"),
            self.project_root / "data/longitudinal.parquet",
        )

    def test_single_merge_uses_current_config_filename_by_default(self) -> None:
        self.config.update(
            dataset="prepared",
            filename="my-study.parquet",
        )
        curator = DataPreprocessor(settings=self.config)
        metadata = pd.DataFrame(
            {FileMetadata.COHORT_Y11_AY: ["2018 - 2019"]}
        )
        captured = {}

        def fake_merge(**kwargs):
            captured.update(kwargs)

        with (
            patch("gleamlet.preprocessing.pipeline.load_dataframe", return_value=metadata),
            patch("gleamlet.preprocessing.pipeline._merge_data", side_effect=fake_merge),
        ):
            curator.merge_data(
                use_file_metadata="metadata.xlsx",
                data_schema={},
                group_by=False,
            )

        self.assertEqual(captured["output_filename"], "my-study.parquet")
        self.assertEqual(
            self.config.get_path("prepared"),
            self.project_root / "data/02_processed/2_merged/my-study.parquet",
        )

    def test_single_merge_helper_honours_configured_filename(self) -> None:
        output_directory = self.project_root / "data/02_processed/2_merged"
        output_directory.mkdir(parents=True)
        destination = output_directory / "longitudinal.parquet"
        items = [
            ("2018-2019", 11, pd.DataFrame({"stud_id": [1], "feature": [2.0]})),
            ("2025-2026", 11, pd.DataFrame({"stud_id": [2], "feature": [3.0]})),
        ]

        with (
            patch(
                "gleamlet.preprocessing.modules._merging_utils.validate_merged_data"
            ),
            patch("gleamlet.preprocessing.modules._merging_utils.print_table"),
            patch("gleamlet.preprocessing.modules._merging_utils.plot_group_heatmap"),
        ):
            merge_one_item_group(
                items,
                output_path=output_directory,
                output_filename=destination.name,
            )

        self.assertTrue(destination.exists())
        self.assertFalse((output_directory / "2018-2026.parquet").exists())


if __name__ == "__main__":
    unittest.main()
