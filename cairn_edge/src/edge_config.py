"""Cairn-Edge configuration loader.

Uses PyYAML when available, but keeps a clear error message so the repo can
still be inspected on minimal systems. Production Jetson images should include
PyYAML in the runtime environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class EdgeNodeConfig:
    raw: Dict[str, Any]

    @property
    def node_id(self) -> str:
        return str(self.raw.get("node", {}).get("id", "cairn-edge-node"))

    @property
    def power_profile(self) -> str:
        return str(self.raw.get("hardware", {}).get("power_profile", "15w_maxn"))

    @property
    def max_tracks(self) -> int:
        return int(self.raw.get("tracking", {}).get("max_tracks", 64))

    @property
    def detector_interval_frames(self) -> int:
        return int(self.raw.get("inference", {}).get("scheduling", {}).get("detector_interval_frames", 3))

    @property
    def air_gapped_default(self) -> bool:
        return bool(self.raw.get("node", {}).get("air_gapped_default", True))


def load_edge_config(path: str | Path) -> EdgeNodeConfig:
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load Cairn-Edge YAML configs. Install with: pip install pyyaml") from exc

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Cairn-Edge config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Cairn-Edge config must be a YAML mapping: {config_path}")

    return EdgeNodeConfig(raw=raw)
