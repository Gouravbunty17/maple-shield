import json
from pathlib import Path

import cv2
import numpy as np

from cairn_edge.src.advanced.thermal_fusion import ThermalFusion, ThermalFusionConfig


def test_replay_rgb_thermal_pairs(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    thermal_path = tmp_path / "thermal.png"
    replay_path = tmp_path / "replay.jsonl"

    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    thermal = np.zeros((60, 80), dtype=np.uint8)
    thermal[15:35, 20:45] = 200
    cv2.imwrite(str(rgb_path), rgb)
    cv2.imwrite(str(thermal_path), thermal)
    replay_path.write_text(json.dumps({"rgb": str(rgb_path), "thermal": str(thermal_path), "expected_min_detections": 1}) + "\n", encoding="utf-8")

    fusion = ThermalFusion(ThermalFusionConfig(thermal_enabled=False, threshold_c=50.0, min_area_px=100))
    for line in replay_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rgb_frame = cv2.imread(row["rgb"])
        thermal_frame = cv2.imread(row["thermal"], cv2.IMREAD_UNCHANGED)
        _, detections = fusion.fuse_with_thermal(rgb_frame, thermal_frame)
        assert len(detections) >= row["expected_min_detections"]


def test_gps_loss_warning_flag_in_klv(tmp_path):
    from cairn_edge.src.advanced.stanag4609_export import PlatformState, STANAG4609Exporter

    exporter = STANAG4609Exporter(tmp_path, platform_state=PlatformState(precision_time_available=False), enabled=False)
    klv = exporter.build_klv_frame([])
    assert klv.warning_flags == ["precision_time_unavailable_system_time_used"]
