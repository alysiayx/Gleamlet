from __future__ import annotations

import unittest

import gleamlet
from gleamlet import DataPreprocessor, FeatureEngineer, GleamletConfig, NEETMLConfig


class PublicApiTests(unittest.TestCase):
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
