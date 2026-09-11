from .config import GleamletConfig, NEETMLConfig

__all__ = [
    "GleamletConfig",
    "DataPreprocessor",
    "FeatureEngineer",
]


def __getattr__(name: str):
    """Load workflow classes only when requested."""
    if name == "DataPreprocessor":
        from .preprocessing import DataPreprocessor

        return DataPreprocessor
    if name == "FeatureEngineer":
        from .features import FeatureEngineer

        return FeatureEngineer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
