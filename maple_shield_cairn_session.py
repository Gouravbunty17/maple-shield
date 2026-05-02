"""CAIRN routing helper for Maple Shield pipelines.

Use this from any Maple Shield script (including ``maple_shield_mvp.py``)
that wants to route detections through ``CairnEngine`` and write
``CairnFrameRecord`` JSONL alongside whatever else the script does. The
helper is intentionally small and side-effect-free except for the JSONL
file you ask it to open.

The helper does NOT replace the existing scoring layer in
``maple_shield_mvp.py`` and does NOT change live behaviour unless the
caller opts in.

Example
-------

    from maple_shield_cairn_session import CairnSession

    with CairnSession(jsonl_path="runs/live/frames.jsonl") as cairn:
        for frame_idx, dets in pipeline:
            # dets is whatever your tracker emits
            cairn_dets = [
                cairn.to_cairn_detection(
                    track_id=d.track_id,
                    label=d.label,
                    confidence=d.confidence,
                    box=d.bbox,
                    frame_w=W, frame_h=H,
                    velocity_px=d.vel,
                    persistence_frames=d.age,
                )
                for d in dets
            ]
            record = cairn.process_frame(frame_idx, cairn_dets, frame_w=W, frame_h=H)
            # record.risks: list of CairnRiskResult with risk_score,
            # threat_level, reasons, recommended_operator_action.

The class restriction in Maple Shield is drone-only. The CAIRN risk
engine already drops non-drone classes. This helper does not weaken that.
"""

from __future__ import annotations

import json
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from cairn_engine import (
    CairnDetection,
    CairnEngine,
    CairnEngineConfig,
    CairnFrameRecord,
)


class CairnSession(AbstractContextManager):
    """Open a CAIRN session and optionally tee CairnFrameRecord rows to JSONL."""

    def __init__(
        self,
        engine: Optional[CairnEngine] = None,
        config: Optional[CairnEngineConfig] = None,
        jsonl_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if engine is not None and config is not None:
            raise ValueError("provide either engine or config, not both")
        self.engine = engine or CairnEngine(config or CairnEngineConfig())
        self.session_id = session_id or f"maple-shield-{int(time.time())}"
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._fh = None

    def __enter__(self) -> "CairnSession":
        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._jsonl_path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    @staticmethod
    def to_cairn_detection(
        *,
        track_id: int,
        label: str,
        confidence: float,
        box: Sequence[int],
        frame_w: int,
        frame_h: int,
        track_confirmed: bool = True,
        velocity_px: float = 0.0,
        vx: float = 0.0,
        vy: float = 0.0,
        persistence_frames: int = 0,
        source: str = "eo_camera",
    ) -> CairnDetection:
        return CairnDetection(
            track_id=int(track_id),
            label=str(label),
            confidence=float(confidence),
            box=[int(b) for b in box],
            frame_w=int(frame_w),
            frame_h=int(frame_h),
            track_confirmed=bool(track_confirmed),
            velocity_px=float(velocity_px),
            vx=float(vx),
            vy=float(vy),
            persistence_frames=int(persistence_frames),
            source=str(source),
        )

    def process_frame(
        self,
        frame_id: int,
        detections: Iterable[CairnDetection],
        *,
        frame_w: int,
        frame_h: int,
        fps: float = 0.0,
        infer_ms: Optional[float] = None,
        health: Optional[dict] = None,
    ) -> CairnFrameRecord:
        record = self.engine.process_frame(
            frame_id=frame_id,
            detections=list(detections),
            session_id=self.session_id,
            frame_w=frame_w,
            frame_h=frame_h,
            fps=fps,
            infer_ms=infer_ms,
            health=health,
        )
        if self._fh is not None:
            self._fh.write(json.dumps(record.to_json(), sort_keys=True) + "\n")
            self._fh.flush()
        return record

    # convenience for callers that prefer a list[dict] of detections
    def process_dict_frame(
        self,
        frame_id: int,
        det_dicts: Iterable[dict],
        *,
        frame_w: int,
        frame_h: int,
        fps: float = 0.0,
        infer_ms: Optional[float] = None,
        health: Optional[dict] = None,
    ) -> CairnFrameRecord:
        cairn_dets: List[CairnDetection] = [
            self.to_cairn_detection(
                track_id=d["track_id"],
                label=d.get("label", "drone"),
                confidence=d.get("confidence", 0.0),
                box=d.get("box") or d.get("bbox"),
                frame_w=frame_w,
                frame_h=frame_h,
                track_confirmed=d.get("track_confirmed", True),
                velocity_px=d.get("velocity_px", 0.0),
                vx=d.get("vx", 0.0),
                vy=d.get("vy", 0.0),
                persistence_frames=d.get("persistence_frames", 0),
                source=d.get("source", "eo_camera"),
            )
            for d in det_dicts
        ]
        return self.process_frame(
            frame_id=frame_id,
            detections=cairn_dets,
            frame_w=frame_w,
            frame_h=frame_h,
            fps=fps,
            infer_ms=infer_ms,
            health=health,
        )
