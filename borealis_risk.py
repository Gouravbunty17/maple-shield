import numpy as np

# Forward sector (fraction of frame width, centered)
FORWARD_SECTOR_WIDTH = 0.6

# Distance proxy thresholds (area ratio)
CLOSE_THRESHOLD = 0.15
MEDIUM_THRESHOLD = 0.05

# Risk weights
WEIGHT_SECTOR = 0.35
WEIGHT_DISTANCE = 0.35
WEIGHT_VERTICAL = 0.15
WEIGHT_CONFIDENCE = 0.15

# State thresholds
UNSAFE_THRESHOLD = 0.70
CAUTION_THRESHOLD = 0.40

def in_forward_sector(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    sector_left = frame_w * (1 - FORWARD_SECTOR_WIDTH) / 2.0
    sector_right = frame_w * (1 + FORWARD_SECTOR_WIDTH) / 2.0
    return sector_left <= cx <= sector_right

def distance_proxy_area(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    box_area = max(1.0, (x2 - x1)) * max(1.0, (y2 - y1))
    area_ratio = box_area / float(frame_w * frame_h)

    if area_ratio >= CLOSE_THRESHOLD:
        return 1.0, area_ratio
    elif area_ratio >= MEDIUM_THRESHOLD:
        return 0.5, area_ratio
    else:
        return 0.0, area_ratio

def vertical_proxy(box, frame_w, frame_h):
    # Bottom of box closer to bottom of frame => likely closer (simple camera geometry proxy)
    x1, y1, x2, y2 = box
    y_bottom = float(y2)
    v = y_bottom / float(max(1, frame_h))   # 0..1
    # Map: below mid-frame increases risk more
    v_score = np.clip((v - 0.45) / (1.0 - 0.45), 0.0, 1.0)
    return float(v_score), float(v)

def compute_risk_score(box, conf, frame_w, frame_h):
    sector = 1.0 if in_forward_sector(box, frame_w, frame_h) else 0.0
    dist_score, area_ratio = distance_proxy_area(box, frame_w, frame_h)
    v_score, v_norm = vertical_proxy(box, frame_w, frame_h)
    conf = float(conf)

    risk = (
        WEIGHT_SECTOR * sector +
        WEIGHT_DISTANCE * dist_score +
        WEIGHT_VERTICAL * v_score +
        WEIGHT_CONFIDENCE * conf
    )
    risk = float(np.clip(risk, 0.0, 1.0))

    if risk >= UNSAFE_THRESHOLD:
        state = "UNSAFE"
    elif risk >= CAUTION_THRESHOLD:
        state = "CAUTION"
    else:
        state = "SAFE"

    return {
        "risk_score": risk,
        "risk_state": state,
        "in_forward_sector": bool(sector),
        "area_ratio": float(area_ratio),
        "vertical_norm": float(v_norm),
        "confidence": conf
    }

def get_risk_color(risk_state):
    if risk_state == "UNSAFE":
        return (0, 0, 255)     # Red
    elif risk_state == "CAUTION":
        return (0, 165, 255)   # Orange
    else:
        return (0, 255, 0)     # Green
