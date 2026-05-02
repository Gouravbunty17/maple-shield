import time

import cv2
import numpy as np

from cairn_edge.src.advanced.thermal_fusion import ThermalFusion, ThermalFusionConfig


def test_thresholding_detects_hotspot_without_camera():
    rgb = np.zeros((120, 160, 3), dtype=np.uint8)
    thermal = np.zeros((60, 80), dtype=np.uint8)
    thermal[20:35, 30:50] = 180  # about 70C using fallback 8-bit scale
    fusion = ThermalFusion(ThermalFusionConfig(thermal_enabled=False, threshold_c=50.0, min_area_px=100))
    fused, detections = fusion.fuse_with_thermal(rgb, thermal)
    assert fused.shape == rgb.shape
    assert len(detections) == 1
    assert detections[0].temperature_c > 50
    assert detections[0].area_px >= 100


def test_homography_warp_with_synthetic_translation():
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    thermal = np.zeros((100, 100), dtype=np.uint8)
    thermal[20:40, 20:40] = 220
    fusion = ThermalFusion(ThermalFusionConfig(thermal_enabled=False, threshold_c=50.0, min_area_px=50))
    fusion.homography = np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1]], dtype=np.float32)
    _, detections = fusion.fuse_with_thermal(rgb, thermal)
    assert detections
    x, y, w, h = detections[0].bbox
    assert x >= 25
    assert y >= 20


def test_thermal_camera_disconnect_reports_degraded():
    fusion = ThermalFusion(ThermalFusionConfig(thermal_enabled=True, camera_id="/dev/does-not-exist"))
    assert fusion.fuse(np.zeros((10, 10, 3), dtype=np.uint8))[1] == []
    status = fusion.health()
    assert status.status == "degraded"
    assert "thermal camera" in (status.degraded_reason or "")
