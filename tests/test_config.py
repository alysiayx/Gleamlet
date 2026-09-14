from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from gleamlet.config import GleamletConfig


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name).resolve()
        self.default_path = self.root / "package/gleamlet/configs/default_config.yaml"
        self.default_path.parent.mkdir(parents=True)
        self.default_path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "project": {"root": "."},
                    "paths": {
                        "data_dir": "data",
                        "processed_dir": {
                            "base": "data_dir",
                            "path": "02_processed",
                        },
                        "cleaned_dir": {
                            "base": "processed_dir",
                            "path": "1_cleaned",
                        },
                        "merged_dir": {
                            "base": "processed_dir",
                            "path": "2_merged",
                        },
                        "aggregated_dir": {
                            "base": "processed_dir",
                            "path": "5_aggregated",
                        },
                        "outputs_dir": "outputs",
                        "logs_dir": "outputs/logs",
                        "models_dir": "outputs/models",
                        "experiments_dir": "outputs/experiments",
                    },
                    "datasets": {
                        "prepared": {
                            "dir": "merged_dir",
                            "file": "longitudinal.parquet",
                        },
                        "model_input": {
                            "dir": "processed_dir",
                            "file": "longitudinal.parquet",
                        }
                    },
                    "profiles": {
                        "sample": {
                            "dataset": "sample",
                            "output": "sample",
                        },
                        "real": {
                            "dataset": "model_input",
                            "output": "real",
                        },
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.project_root = self.root / "study"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def load(self, *, user_config_path: Path | None = None) -> GleamletConfig:
        return GleamletConfig.load(
            project_root=self.project_root,
            default_config_path=self.default_path,
            user_config_path=user_config_path,
        )

    def test_default_paths_resolve_from_project_root(self) -> None:
        settings = self.load()
        self.assertEqual(
            settings.get_path("processed_dir"),
            self.project_root / "data/02_processed",
        )
        self.assertEqual(
            settings.get_path("prepared"),
            self.project_root / "data/02_processed/2_merged/longitudinal.parquet",
        )
        self.assertEqual(
            settings.get_path("model_input"),
            self.project_root / "data/02_processed/longitudinal.parquet",
        )
        self.assertEqual(
            settings.get_path("cleaned_dir"),
            self.project_root / "data/02_processed/1_cleaned",
        )
        self.assertEqual(
            settings.get_path("logs_dir"),
            self.project_root / "outputs/logs",
        )
        self.assertEqual(
            settings.get_path("models_dir"),
            self.project_root / "outputs/models",
        )

    def test_unknown_registry_names_are_rejected(self) -> None:
        settings = self.load()

        with self.assertRaises(KeyError):
            settings.get_path("processed_data_dir")
        with self.assertRaises(KeyError):
            settings.get_path("real")

    def test_stage_paths_follow_configured_processed_directory(self) -> None:
        settings = self.load()
        settings.update(paths={"processed_dir": "outputs/processed"})

        self.assertEqual(
            settings.get_path("cleaned_dir"),
            self.project_root / "outputs/processed/1_cleaned",
        )
        self.assertEqual(
            settings.get_path("merged_dir"),
            self.project_root / "outputs/processed/2_merged",
        )

    def test_data_directory_relocates_all_data_stages(self) -> None:
        settings = GleamletConfig.load(project_root=self.project_root)
        settings.update(data_dir="/secure/neet-data")

        self.assertEqual(
            settings.get_path("raw_dir"),
            Path("/secure/neet-data/00_source"),
        )
        self.assertEqual(
            settings.get_path("processed_dir"),
            Path("/secure/neet-data/02_processed"),
        )
        self.assertEqual(
            settings.get_path("prepared"),
            Path("/secure/neet-data/02_processed/2_merged/longitudinal.parquet"),
        )
        self.assertEqual(
            settings.get_path("model_input"),
            Path("/secure/neet-data/longitudinal.parquet"),
        )

    def test_stage_path_can_be_overridden_independently(self) -> None:
        settings = self.load()
        settings.update(paths={"cleaned_dir": "/secure/cleaned"})

        self.assertEqual(
            settings.get_path("cleaned_dir"),
            Path("/secure/cleaned"),
        )

    def test_prepared_dataset_can_use_a_different_preparation_stage(self) -> None:
        settings = self.load()
        settings.update(dataset="prepared", folder="aggregated_dir")

        self.assertEqual(
            settings.get_path("prepared"),
            self.project_root
            / "data/02_processed/5_aggregated/longitudinal.parquet",
        )

    def test_user_config_overrides_defaults(self) -> None:
        user_path = self.root / "user.yaml"
        user_path.write_text(
            "datasets:\n  model_input:\n    file: 2018-2026.parquet\n",
            encoding="utf-8",
        )
        settings = self.load(user_config_path=user_path)
        self.assertEqual(
            settings.get_path("model_input").name,
            "2018-2026.parquet",
        )
        self.assertEqual(
            settings.config["datasets"]["model_input"]["file"],
            "2018-2026.parquet",
        )

    def test_environment_overrides_user_config(self) -> None:
        user_path = self.root / "user.yaml"
        user_path.write_text(
            "datasets:\n  model_input:\n    file: user.parquet\n",
            encoding="utf-8",
        )
        with patch.dict(
            os.environ,
            {"GLEAMLET_MODEL_INPUT_PATH": "/secure/environment.parquet"},
        ):
            settings = self.load(user_config_path=user_path)
            self.assertEqual(
                settings.get_path("model_input"),
                Path("/secure/environment.parquet"),
            )

    def test_explicit_path_overrides_environment(self) -> None:
        with patch.dict(
            os.environ,
            {"GLEAMLET_MODEL_INPUT_PATH": "/secure/environment.parquet"},
        ):
            settings = self.load()
            self.assertEqual(
                settings.get_path(
                    "model_input", path="explicit.parquet"
                ),
                self.project_root / "explicit.parquet",
            )

    def test_settings_persist_only_user_overrides(self) -> None:
        user_path = self.root / ".gleamlet/config.yaml"
        settings = self.load(user_config_path=user_path)
        settings.update(
            dataset="model_input",
            folder="/secure/neet",
            filename="2018-2026.parquet",
        )
        settings.save()
        saved = yaml.safe_load(user_path.read_text(encoding="utf-8"))
        self.assertEqual(
            saved,
            {
                "datasets": {
                    "model_input": {
                        "dir": "/secure/neet",
                        "file": "2018-2026.parquet",
                    }
                }
            },
        )

    def test_profile_outputs_are_distinct(self) -> None:
        settings = GleamletConfig.load(project_root=self.project_root)
        artifact_root = settings.get_path("experiments_dir")
        sample_dir = artifact_root / settings.profile("sample")["output"]
        real_dir = artifact_root / settings.profile("real")["output"]
        self.assertNotEqual(sample_dir, real_dir)

    def test_environment_project_root_overrides_user_config(self) -> None:
        user_path = self.root / "user.yaml"
        user_path.write_text(
            "project:\n  root: /configured/project\n",
            encoding="utf-8",
        )
        environment_root = self.root / "environment-project"
        with patch.dict(os.environ, {"GLEAMLET_PROJECT_ROOT": str(environment_root)}):
            settings = GleamletConfig.load(
                default_config_path=self.default_path,
                user_config_path=user_path,
            )
        self.assertEqual(settings.project_root, environment_root.resolve())

    def test_legacy_environment_variable_remains_supported(self) -> None:
        environment_root = self.root / "legacy-environment-project"
        with patch.dict(
            os.environ,
            {"NEETML_PROJECT_ROOT": str(environment_root)},
            clear=True,
        ):
            settings = GleamletConfig.load(default_config_path=self.default_path)

        self.assertEqual(settings.project_root, environment_root.resolve())

    def test_legacy_user_config_is_used_when_new_path_is_absent(self) -> None:
        legacy_path = self.project_root / ".neetml/config.yaml"
        legacy_path.parent.mkdir(parents=True)
        legacy_path.write_text(
            "datasets:\n  model_input:\n    file: legacy.parquet\n",
            encoding="utf-8",
        )

        settings = GleamletConfig.load(
            project_root=self.project_root,
            default_config_path=self.default_path,
        )

        self.assertEqual(settings.user_config_path, legacy_path)
        self.assertEqual(settings.get_path("model_input").name, "legacy.parquet")

    def test_project_root_update_is_persisted(self) -> None:
        user_path = self.root / ".gleamlet/config.yaml"
        new_root = self.root / "relocated-study"
        settings = self.load(user_config_path=user_path)
        settings.update(project_root=new_root).save()

        reloaded = GleamletConfig.load(
            default_config_path=self.default_path,
            user_config_path=user_path,
        )
        self.assertEqual(reloaded.project_root, new_root.resolve())

    def test_year_range_naming_updates_and_persists_registry(self) -> None:
        user_path = self.root / ".gleamlet/config.yaml"
        settings = self.load(user_config_path=user_path)

        settings.update(
            dataset="model_input",
            year_range=(2018, 2026),
        ).save()
        configured_path = settings.get_path("model_input")
        reloaded = self.load(user_config_path=user_path)

        self.assertEqual(configured_path.name, "2018-2026.parquet")
        self.assertEqual(
            reloaded.get_path("model_input").name,
            "2018-2026.parquet",
        )

    def test_stable_longitudinal_name_is_used_without_year_range(self) -> None:
        user_path = self.root / ".gleamlet/config.yaml"
        settings = self.load(user_config_path=user_path)

        configured_path = settings.get_path("model_input")

        self.assertEqual(configured_path.name, "longitudinal.parquet")

    def test_longitudinal_naming_rejects_ambiguous_or_non_parquet_input(self) -> None:
        settings = self.load(user_config_path=self.root / ".gleamlet/config.yaml")

        with self.assertRaises(ValueError):
            settings.update(
                dataset="model_input",
                filename="custom.parquet",
                year_range=(2018, 2026),
            )
        with self.assertRaises(ValueError):
            settings.update(dataset="model_input", filename="custom.csv")

    def test_reset_removes_user_config_and_restores_defaults(self) -> None:
        user_path = self.root / ".gleamlet/config.yaml"
        settings = self.load(user_config_path=user_path)
        settings.update(
            dataset="model_input",
            filename="2018-2026.parquet",
        ).save()

        settings.reset()

        self.assertFalse(user_path.exists())
        self.assertEqual(
            settings.get_path("model_input").name,
            "longitudinal.parquet",
        )

    def test_config_changes_are_logged(self) -> None:
        user_path = self.root / ".gleamlet/config.yaml"
        settings = self.load(user_config_path=user_path)

        with self.assertLogs("gleamlet.config", level="INFO") as logs:
            settings.update(data_dir="/secure/data").save()
            settings.reset()

        messages = "\n".join(logs.output)
        self.assertIn("data_dir=/secure/data", messages)
        self.assertIn(f"Saved user configuration: {user_path}", messages)
        self.assertIn("Reset user configuration", messages)

    def test_get_path_reports_resolved_configuration(self) -> None:
        settings = self.load()

        table = settings.get_path(
            keys=["project_root", "processed_dir", "model_input"]
        )
        paths = dict(zip(table["Type"], table["Path"]))

        self.assertEqual(paths["project_root"], self.project_root)
        self.assertEqual(
            paths["processed_dir"],
            self.project_root / "data/02_processed",
        )
        self.assertEqual(
            paths["model_input"],
            self.project_root / "data/02_processed/longitudinal.parquet",
        )
        self.assertIn(
            "model_input",
            settings.get_path(keys="all")["Type"].values,
        )


if __name__ == "__main__":
    unittest.main()
