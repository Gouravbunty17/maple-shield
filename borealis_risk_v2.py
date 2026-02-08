import numpy as np

# -----------------------
# Risk Scoring Config (V2)
# -----------------------
FORWARD_SECTOR_WIDTH = 0.6

CLOSE_THRESHOLD = 0.15
MEDIUM_THRESHOLD = 0.05

# weights sum = 1.0
WEIGHT_SECTOR = 0.25
WEIGHT_DISTANCE = 0.25
WEIGHT_CONFIDENCE = 0.15
WEIGHT_VELOCITY = 0.20
WEIGHT_HEADING = 0.15

UNSAFE_THRESHOLD = 0.7
CAUTION_THRESHOLD = 0.4

# Velocity thresholds (pixels/frame)
VELOCITY_HIGH = 15.0
VELOCITY_MED = 5.0

# Persistence escalation
PERSISTENCE_FRAMES = 30

def in_forward_sector(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    sector_left = frame_w * (1 - FORWARD_SECTOR_WIDTH) / 2.0
    sector_right = frame_w * (1 + FORWARD_SECTOR_WIDTH) / 2.0
    return sector_left <= cx <= sector_right

def distance_proxy(box, frame_w, frame_h):
    x1, y1, x2, y2 = box
    box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    frame_area = float(frame_w * frame_h)
    area_ratio = box_area / (frame_area + 1e-6)

    if area_ratio >= CLOSE_THRESHOLD:
        return 1.0
    elif area_ratio >= MEDIUM_THRESHOLD:
        return 0.5
    else:
        return 0.0

def velocity_score(velocity_data):
    speed = float((velocity_data or {}).get("speed", 0.0))
    if speed >= VELOCITY_HIGH:
        return 1.0
    elif speed >= VELOCITY_MED:
        return 0.5
    else:
        return 0.0

def heading_score(velocity_data, box, frame_w, frame_h):
    vel = velocity_data or {}
    vx = float(vel.get("vx", 0.0))
    vy = float(vel.get("vy", 0.0))

    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    frame_cx = frame_w / 2.0
    frame_cy = frame_h / 2.0

    to_center_x = frame_cx - cx
    to_center_y = frame_cy - cy

    dist = np.sqrt(to_center_x**2 + to_center_y**2)
    if dist < 1e-6:
        return 0.0

    dot = vx * to_center_x + vy * to_center_y
    approach = dot / dist

    # Map to [0,1]
    if approach > 5.0:
        return 1.0
    elif approach > 0.0:
        return float(approach / 5.0)
    else:
        return 0.0

def compute_risk_score(box, conf, frame_w, frame_h, velocity_data=None, persistence_count=0):
    in_sector = 1.0 if in_forward_sector(box, frame_w, frame_h) else 0.0
    dist_score = distance_proxy(box, frame_w, frame_h)
    conf_score = float(conf)

    vel_score = velocity_score(velocity_data)
    head_score = heading_score(velocity_data, box, frame_w, frame_h)

    risk = (
        WEIGHT_SECTOR * in_sector +
        WEIGHT_DISTANCE * dist_score +
        WEIGHT_CONFIDENCE * conf_score +
        WEIGHT_VELOCITY * vel_score +
        WEIGHT_HEADING * head_score
    )

    # Persistence escalation (adds up to +0.2)
    if int(persistence_count) >= PERSISTENCE_FRAMES:
        escalation = min(0.2, (int(persistence_count) - PERSISTENCE_FRAMES) * 0.01)
        risk = min(1.0, risk + escalation)

    risk = float(np.clip(risk, 0.0, 1.0))

    if risk >= UNSAFE_THRESHOLD:
        state = "UNSAFE"
    elif risk >= CAUTION_THRESHOLD:
        state = "CAUTION"
    else:
        state = "SAFE"

    dist_cat = "CLOSE" if dist_score == 1.0 else ("MEDIUM" if dist_score == 0.5 else "FAR")

    return {
        "risk_score": risk,
        "risk_state": state,
        "in_forward_sector": bool(in_sector),
        "distance_category": dist_cat,
        "confidence": float(conf),
        "velocity_score": float(vel_score),
        "heading_score": float(head_score),
        "persistence_frames": int(persistence_count),
    }

def get_risk_color(risk_state):
    if risk_state == "UNSAFE":
        return (0, 0, 255)
    elif risk_state == "CAUTION":
        return (0, 165, 255)
    else:
        return (0, 255, 0)
