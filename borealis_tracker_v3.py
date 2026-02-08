import numpy as np
from collections import deque

TRACK_IOU_THRES = 0.30
TRACK_MAX_AGE = 30
TRACK_MIN_HITS = 2
TRACK_SMOOTH_ALPHA = 0.70
VELOCITY_HISTORY_LENGTH = 5

def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / (area_a + area_b - inter + 1e-6)

class Track:
    def __init__(self, track_id, box, cls, conf, label, frame_id, timestamp):
        self.id = int(track_id)
        self.box = np.array(box, dtype=np.float32)
        self.cls = int(cls)
        self.conf = float(conf)
        self.label = str(label)

        self.hits = 1
        self.age = 0
        self.confirmed = False

        self.birth_frame = int(frame_id)
        self.birth_timestamp = float(timestamp)
        self.last_seen_frame = int(frame_id)
        self.last_seen_timestamp = float(timestamp)

        self.death_frame = None
        self.death_timestamp = None

        self.position_history = deque(maxlen=VELOCITY_HISTORY_LENGTH)
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        self.position_history.append((cx, cy, int(frame_id)))

        self.conf_sum = float(conf)
        self.conf_count = 1

        # Persistence: consecutive frames with CAUTION/UNSAFE
        self.high_risk_count = 0

    def update(self, box, conf, frame_id, timestamp, alpha=TRACK_SMOOTH_ALPHA):
        box = np.array(box, dtype=np.float32)
        self.box = alpha * self.box + (1.0 - alpha) * box
        self.conf = float(conf)

        self.hits += 1
        self.age = 0

        self.last_seen_frame = int(frame_id)
        self.last_seen_timestamp = float(timestamp)

        if self.hits >= TRACK_MIN_HITS:
            self.confirmed = True

        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        self.position_history.append((cx, cy, int(frame_id)))

        self.conf_sum += float(conf)
        self.conf_count += 1

    def mark_missed(self):
        self.age += 1

    def kill(self, frame_id, timestamp):
        self.death_frame = int(frame_id)
        self.death_timestamp = float(timestamp)

    def get_velocity(self):
        if len(self.position_history) < 2:
            return 0.0, 0.0, 0.0
        (x0, y0, f0) = self.position_history[0]
        (x1, y1, f1) = self.position_history[-1]
        df = f1 - f0
        if df <= 0:
            return 0.0, 0.0, 0.0
        vx = (x1 - x0) / df
        vy = (y1 - y0) / df
        speed = float(np.sqrt(vx * vx + vy * vy))
        return float(vx), float(vy), speed

    def update_risk_state(self, risk_state):
        if risk_state in ("CAUTION", "UNSAFE"):
            self.high_risk_count += 1
        else:
            self.high_risk_count = 0

    def get_persistence_count(self):
        return int(self.high_risk_count)

    def get_avg_confidence(self):
        return float(self.conf_sum / self.conf_count) if self.conf_count > 0 else 0.0

    def to_lifecycle_dict(self):
        vx, vy, speed = self.get_velocity()
        return {
            "track_id": self.id,
            "class": self.cls,
            "label": self.label,
            "birth_frame": self.birth_frame,
            "birth_timestamp": self.birth_timestamp,
            "death_frame": self.death_frame,
            "death_timestamp": self.death_timestamp,
            "lifetime_frames": int(self.hits),
            "lifetime_seconds": (self.death_timestamp - self.birth_timestamp) if self.death_timestamp else None,
            "avg_confidence": self.get_avg_confidence(),
            "final_velocity_px_per_frame": {"vx": vx, "vy": vy, "speed": speed},
            "max_persistence": int(self.high_risk_count),
        }

class IoUTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.lifecycle_events = []

    def step(self, det_list, frame_id, timestamp):
        for t in self.tracks.values():
            t.mark_missed()

        dets = sorted(det_list, key=lambda d: float(d.get("conf", 0.0)), reverse=True)
        assigned = set()

        for d in dets:
            best_tid = None
            best_iou = 0.0

            for tid, t in self.tracks.items():
                if tid in assigned:
                    continue
                if t.cls != int(d["cls"]):
                    continue
                i = iou_xyxy(t.box.tolist(), d["box"])
                if i > best_iou:
                    best_iou = i
                    best_tid = tid

            if best_tid is not None and best_iou >= TRACK_IOU_THRES:
                t = self.tracks[best_tid]
                t.update(d["box"], d["conf"], frame_id, timestamp)
                assigned.add(best_tid)

                vx, vy, speed = t.get_velocity()
                d["track_id"] = int(t.id)
                d["track_confirmed"] = bool(t.confirmed)
                d["velocity"] = {"vx": vx, "vy": vy, "speed": speed}
                d["persistence_count"] = t.get_persistence_count()
            else:
                tid = self.next_id
                self.next_id += 1
                t = Track(tid, d["box"], d["cls"], d["conf"], d["label"], frame_id, timestamp)
                self.tracks[tid] = t
                assigned.add(tid)

                d["track_id"] = int(t.id)
                d["track_confirmed"] = bool(t.confirmed)
                d["velocity"] = {"vx": 0.0, "vy": 0.0, "speed": 0.0}
                d["persistence_count"] = 0

        dead = []
        for tid, t in self.tracks.items():
            if t.age > TRACK_MAX_AGE:
                t.kill(frame_id, timestamp)
                self.lifecycle_events.append(t.to_lifecycle_dict())
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

        return det_list

    def update_risk_states(self, det_list):
        for d in det_list:
            tid = d.get("track_id", None)
            if tid is None:
                continue
            t = self.tracks.get(int(tid))
            if t is None:
                continue
            risk_state = (d.get("risk", {}) or {}).get("risk_state", "SAFE")
            t.update_risk_state(risk_state)

    def get_lifecycle_summary(self):
        return {
            "total_tracks_created": int(self.next_id - 1),
            "active_tracks": int(len(self.tracks)),
            "dead_tracks": int(len(self.lifecycle_events)),
            "lifecycle_events": self.lifecycle_events,
        }
