"""YOLO output provider for CAIRN detections.

The provider accepts an ONNX-like session, runs YOLO-style inference, then
returns only drone-labeled ``CairnDetection`` rows. Non-drone rows are dropped
before any CAIRN object is built.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    repo_root_s = str(repo_root)
    if repo_root_s not in sys.path:
        sys.path.insert(0, repo_root_s)


_ensure_repo_root_on_path()

from cairn_engine import CairnDetection  # noqa: E402


ALLOWED_CLASSES = frozenset({"drone"})
COCO80 = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush"
).split()


class OnnxInput(Protocol):
    name: str


class OnnxLikeSession(Protocol):
    def run(self, output_names: Optional[Sequence[str]], input_feed: dict): ...
    def get_inputs(self) -> Sequence[OnnxInput]: ...


class YoloDetectionProvider:
    """Convert YOLO output tensors into drone-only CAIRN detections."""

    def __init__(
        self,
        session: OnnxLikeSession,
        *,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        max_detections: int = 50,
        class_names: Sequence[str] = COCO80,
        allowed_classes: Iterable[str] = ALLOWED_CLASSES,
        track_id_factory: Optional[Callable[[], int]] = None,
    ):
        self.session = session
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.max_detections = int(max_detections)
        self.class_names = [name.strip().lower() for name in class_names]
        self.allowed_classes = frozenset(name.strip().lower() for name in allowed_classes)
        self._next_track_id = 1
        self._track_id_factory = track_id_factory or self._default_track_id
        self._tracks: dict[int, Tuple[float, float, float, float]] = {}
        self._track_counts: dict[int, int] = {}
        self._track_centers: dict[int, Tuple[float, float]] = {}

    @classmethod
    def from_onnx(cls, model_path: str | Path, **kwargs) -> "YoloDetectionProvider":
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Install onnxruntime to load a YOLO ONNX model.") from e
        return cls(ort.InferenceSession(str(model_path)), **kwargs)

    def __call__(self, frame_idx: int, frame, frame_w: int, frame_h: int) -> List[CairnDetection]:
        if frame is None:
            return []

        raw_outputs = self.session.run(None, {self._input_name(): self._preprocess(frame)})
        rows = self._postprocess(raw_outputs[0], frame_w, frame_h)
        detections: List[CairnDetection] = []
        for row in rows:
            label = row["label"]
            if label not in self.allowed_classes:
                continue
            track_id, persistence, velocity_px, vx, vy = self._assign_track(row["bbox"])
            detections.append(
                CairnDetection(
                    track_id=track_id,
                    label=label,
                    confidence=float(row["confidence"]),
                    box=[int(round(v)) for v in row["bbox"]],
                    frame_w=frame_w,
                    frame_h=frame_h,
                    track_confirmed=persistence > 1,
                    velocity_px=velocity_px,
                    vx=vx,
                    vy=vy,
                    persistence_frames=persistence,
                    source="yolo_onnx",
                    raw={"frame_idx": frame_idx, "provider": "YoloDetectionProvider"},
                )
            )
        return detections

    def _default_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _input_name(self) -> str:
        inputs = self.session.get_inputs()
        return inputs[0].name if inputs else "images"

    def _preprocess(self, frame) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.ndim != 3:
            raise ValueError("YOLO provider expects an HxWxC frame.")
        arr = arr / 255.0
        return np.transpose(arr, (2, 0, 1))[None, ...]

    def _postprocess(self, output, frame_w: int, frame_h: int) -> list[dict]:
        tensor = np.asarray(output, dtype=np.float32)
        rows = self._normalize_yolo_tensor(tensor)
        candidates: list[dict] = []
        for row in rows:
            if row.shape[0] < 5:
                continue
            cls_scores = row[4:]
            if cls_scores.size == 0:
                continue
            cls_idx = int(np.argmax(cls_scores))
            confidence = float(cls_scores[cls_idx])
            if confidence < self.conf_threshold:
                continue
            label = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}"
            candidates.append({
                "label": label,
                "confidence": confidence,
                "bbox": self._xywh_to_xyxy(row[:4], frame_w, frame_h),
            })

        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        keep: list[dict] = []
        for cand in candidates:
            if len(keep) >= self.max_detections:
                break
            if all(_iou(cand["bbox"], prev["bbox"]) < self.iou_threshold for prev in keep):
                keep.append(cand)
        return keep

    def _normalize_yolo_tensor(self, tensor: np.ndarray) -> np.ndarray:
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim != 2:
            raise ValueError(f"Unsupported YOLO output shape: {tensor.shape}")
        if tensor.shape[0] >= 5 and tensor.shape[0] < tensor.shape[1]:
            tensor = tensor.T
        return tensor

    def _xywh_to_xyxy(self, xywh, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
        cx, cy, w, h = (float(v) for v in xywh)
        x1 = max(0.0, cx - w / 2.0)
        y1 = max(0.0, cy - h / 2.0)
        x2 = min(float(frame_w), cx + w / 2.0)
        y2 = min(float(frame_h), cy + h / 2.0)
        return x1, y1, x2, y2

    def _assign_track(self, bbox: Tuple[float, float, float, float]):
        best_id = None
        best_iou = 0.0
        for track_id, old_bbox in self._tracks.items():
            score = _iou(bbox, old_bbox)
            if score > best_iou:
                best_id = track_id
                best_iou = score

        track_id = best_id if best_id is not None and best_iou >= self.iou_threshold else self._track_id_factory()
        old_center = self._track_centers.get(track_id, _center(bbox))
        new_center = _center(bbox)
        vx = new_center[0] - old_center[0]
        vy = new_center[1] - old_center[1]
        velocity_px = float((vx * vx + vy * vy) ** 0.5)

        self._tracks[track_id] = bbox
        self._track_centers[track_id] = new_center
        self._track_counts[track_id] = self._track_counts.get(track_id, 0) + 1
        return track_id, self._track_counts[track_id], velocity_px, float(vx), float(vy)


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0
