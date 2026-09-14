from __future__ import annotations

import inspect
import unittest

import gleamlet
import gleamlet.features as features
import gleamlet.preprocessing as preprocessing
from gleamlet import DataPreprocessor, FeatureEngineer, GleamletConfig, NEETMLConfig


class PublicApiTests(unittest.TestCase):
    def test_all_public_callables_accept_verbose(self) -> None:
        missing = []
        checked = set()

        for module in (gleamlet, preprocessing, features):
            for export_name in module.__all__:
                public_callable = getattr(module, export_name)
                if public_callable in checked:
                    continue
                checked.add(public_callable)
                if "verbose" not in inspect.signature(public_callable).parameters:
                    missing.append(export_name)
                if not inspect.isclass(public_callable):
                    continue
                for name, attribute in vars(public_callable).items():
                    if name.startswith("_") or isinstance(attribute, property):
                        continue
                    member = getattr(public_callable, name)
                    if (
                        callable(member)
                        and "verbose" not in inspect.signature(member).parameters
                    ):
                        missing.append(f"{export_name}.{name}")

        self.assertEqual(missing, [])

    def test_primary_classes_are_available_from_package_root(self) -> None:
        self.assertEqual(DataPreprocessor.__name__, "DataPreprocessor")
        self.assertEqual(FeatureEngineer.__name__, "FeatureEngineer")
        self.assertEqual(GleamletConfig.__name__, "GleamletConfig")
        self.assertIs(NEETMLConfig, GleamletConfig)

    def test_public_api_matches_current_release_scope(self) -> None:
        self.assertEqual(
            gleamlet.__all__,
            ["GleamletConfig", "DataPreprocessor", "FeatureEngineer"],
        )


if __name__ == "__main__":
    unittest.main()
