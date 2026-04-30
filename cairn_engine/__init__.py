"""CAIRN detection engine package for Maple Shield."""

from .config import CairnEngineConfig, CairnRiskConfig
from .engine import CairnEngine
from .risk_engine import CairnRiskEngine
from .schemas import CairnDetection, CairnFrameRecord, CairnRiskResult, CairnThreatLevel

__all__ = [
    "CairnDetection",
    "CairnEngine",
    "CairnEngineConfig",
    "CairnFrameRecord",
    "CairnRiskConfig",
    "CairnRiskEngine",
    "CairnRiskResult",
    "CairnThreatLevel",
]

__version__ = "2.0.0-dev"
