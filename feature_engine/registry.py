"""Feature registry for schema validation and discoverability."""

from dataclasses import dataclass
from feature_engine.config import FEATURES


@dataclass
class FeatureMetadata:
    name: str
    dtype: str
    source: str
    window: str | None
    default: float | int


class FeatureRegistry:
    """Central catalog of all features with validation."""

    def __init__(self):
        self._features: dict[str, FeatureMetadata] = {}
        self._load_from_config()

    def _load_from_config(self):
        for name, meta in FEATURES.items():
            self._features[name] = FeatureMetadata(
                name=name,
                dtype=meta["type"],
                source=meta["source"],
                window=meta.get("window"),
                default=meta["default"],
            )

    def get(self, name: str) -> FeatureMetadata | None:
        return self._features.get(name)

    def list_features(self, source: str | None = None) -> list[FeatureMetadata]:
        if source:
            return [f for f in self._features.values() if f.source == source]
        return list(self._features.values())

    def validate_feature_vector(self, features: dict) -> list[str]:
        """Return list of missing or invalid features."""
        errors = []
        for name, meta in self._features.items():
            if name not in features:
                errors.append(f"Missing feature: {name}")
            elif meta.dtype == "float" and not isinstance(features[name], (int, float)):
                errors.append(f"Invalid type for {name}: expected float, got {type(features[name])}")
            elif meta.dtype == "int" and not isinstance(features[name], int):
                errors.append(f"Invalid type for {name}: expected int, got {type(features[name])}")
        return errors

    def get_defaults(self) -> dict:
        """Get default values for all features."""
        return {name: meta.default for name, meta in self._features.items()}
