"""
CAIRN Engine wrapper.

The wrapper accepts normalized detections from any detector/tracker pipeline and
returns a frame-level record. Existing Maple Shield scripts can adopt this with
minimal changes, while future sensor fusion can use the same contract.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, Optional
import time

from .config import CairnEngineConfig
from .risk_engine import CairnRiskEngine
from .schemas import CairnDetection, CairnFrameRecord, CairnThreatLevel


class CairnEngine:
    def __init__(self, config: Optional[CairnEngineConfig] = None):
        self.config = config or CairnEngineConfig()
        self.risk_engine = CairnRiskEngine(self.config.risk)
        self.started_ts = time.time()
        self.frames_processed = 0

    def process_frame(
        self,
        frame_id: int,
        detections: Iterable[CairnDetection],
        session_id: str,
        frame_w: int,
        frame_h: int,
        fps: float = 0.0,
        infer_ms: Optional[float] = None,
        health: Optional[Dict[str, Any]] = None,
        ts: Optional[float] = None,
    ) -> CairnFrameRecord:
        self.frames_processed += 1
        risks = self.risk_engine.score_many(detections)

        if risks:
            max_risk = max(risks, key=lambda x: x.risk_score)
            max_level = max_risk.threat_level
            max_score = max_risk.risk_score
        else:
            max_level = CairnThreatLevel.CLEAR
            max_score = 0.0

        runtime_s = time.time() - self.started_ts
        health_payload = {
            "engine": self.config.engine_name,
            "engine_version": self.config.engine_version,
            "runtime_s": round(runtime_s, 3),
            "frames_processed": self.frames_processed,
        }
        if health:
            health_payload.update(health)

        return CairnFrameRecord(
            frame=frame_id,
            ts=time.time() if ts is None else ts,
            session_id=session_id,
            frame_w=frame_w,
            frame_h=frame_h,
            fps=fps,
            infer_ms=infer_ms,
            max_threat_level=max_level,
            max_risk_score=max_score,
            risks=risks,
            health=health_payload,
        )

    def metadata(self) -> Dict[str, Any]:
        return {
            "engine": self.config.engine_name,
            "engine_version": self.config.engine_version,
            "risk_config": asdict(self.config.risk),
            "started_ts": self.started_ts,
        }
