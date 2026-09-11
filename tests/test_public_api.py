from __future__ import annotations

import unittest

import neetml
from neetml import DataPreprocessor, FeatureEngineer, NEETMLConfig


class PublicApiTests(unittest.TestCase):
    def test_primary_classes_are_available_from_package_root(self) -> None:
        self.assertEqual(DataPreprocessor.__name__, "DataPreprocessor")
        self.assertEqual(FeatureEngineer.__name__, "FeatureEngineer")
        self.assertEqual(NEETMLConfig.__name__, "NEETMLConfig")

    def test_public_api_matches_current_release_scope(self) -> None:
        self.assertEqual(
            neetml.__all__,
            ["NEETMLConfig", "DataPreprocessor", "FeatureEngineer"],
        )


if __name__ == "__main__":
    unittest.main()
