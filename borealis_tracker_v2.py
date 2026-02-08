import numpy as np
from collections import deque

TRACK_IOU_THRES = 0.30
TRACK_MAX_AGE = 30
TRACK_MIN_HITS = 2
TRACK_SMOOTH_ALPHA = 0.70
VELOCITY_HISTORY_LENGTH = 5  # frames to keep for velocity estimation

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

        # Lifecycle
        self.birth_frame = int(frame_id)
        self.birth_timestamp = float(timestamp)
        self.last_seen_frame = int(frame_id)
        self.last_seen_timestamp = float(timestamp)
        self.death_frame = None
        self.death_timestamp = None

        # Velocity tracking (center points)
        self.position_history = deque(maxlen=VELOCITY_HISTORY_LENGTH)
        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        self.position_history.append((float(cx), float(cy), int(frame_id)))

        # Stats
        self.conf_sum = float(conf)
        self.conf_count = 1

    def update(self, box, conf, frame_id, timestamp, alpha=TRACK_SMOOTH_ALPHA):
        box = np.array(box, dtype=np.float32)
        self.box = alpha * self.box + (1 - alpha) * box
        self.conf = float(conf)
        self.hits += 1
        self.age = 0
        self.last_seen_frame = int(frame_id)
        self.last_seen_timestamp = float(timestamp)

        if self.hits >= TRACK_MIN_HITS:
            self.confirmed = True

        cx = (self.box[0] + self.box[2]) / 2.0
        cy = (self.box[1] + self.box[3]) / 2.0
        self.position_history.append((float(cx), float(cy), int(frame_id)))

        self.conf_sum += float(conf)
        self.conf_count += 1

    def mark_missed(self):
        self.age += 1

    def kill(self, frame_id, timestamp):
        self.death_frame = int(frame_id)
        self.death_timestamp = float(timestamp)

    def get_velocity(self):
        """
        Velocity in pixels/frame using first and last samples in history.
        Returns (vx, vy, speed)
        """
        if len(self.position_history) < 2:
            return 0.0, 0.0, 0.0

        x0, y0, f0 = self.position_history[0]
        x1, y1, f1 = self.position_history[-1]
        frame_delta = f1 - f0
        if frame_delta <= 0:
            return 0.0, 0.0, 0.0

        vx = (x1 - x0) / frame_delta
        vy = (y1 - y0) / frame_delta
        speed = float(np.sqrt(vx * vx + vy * vy))
        return float(vx), float(vy), float(speed)

    def get_avg_confidence(self):
        return float(self.conf_sum / self.conf_count) if self.conf_count > 0 else 0.0

    def get_lifetime_frames(self):
        # lifetime based on frames between birth and last seen (more meaningful than hits)
        return int(self.last_seen_frame - self.birth_frame + 1)

    def to_lifecycle_dict(self):
        vx, vy, speed = self.get_velocity()
        return {
            "track_id": self.id,
            "cls": self.cls,
            "label": self.label,
            "birth_frame": self.birth_frame,
            "birth_timestamp": self.birth_timestamp,
            "last_seen_frame": self.last_seen_frame,
            "last_seen_timestamp": self.last_seen_timestamp,
            "death_frame": self.death_frame,
            "death_timestamp": self.death_timestamp,
            "confirmed": bool(self.confirmed),
            "hits": int(self.hits),
            "lifetime_frames": self.get_lifetime_frames(),
            "lifetime_seconds": (self.death_timestamp - self.birth_timestamp) if self.death_timestamp is not None else None,
            "avg_confidence": self.get_avg_confidence(),
            "final_velocity_px_per_frame": {"vx": vx, "vy": vy, "speed": speed},
        }

class IoUTracker:
    def __init__(self):
        self.tracks = {}          # tid -> Track
        self.next_id = 1
        self.lifecycle_events = []  # dead tracks exported dicts

    def step(self, det_list, frame_id, timestamp):
        # mark all tracks missed
        for t in self.tracks.values():
            t.mark_missed()

        dets = sorted(det_list, key=lambda d: d.get("conf", 0.0), reverse=True)
        assigned_tracks = set()

        for d in dets:
            best_tid = None
            best_iou = 0.0

            for tid, t in self.tracks.items():
                if tid in assigned_tracks:
                    continue
                if int(d["cls"]) != t.cls:
                    continue
                i = iou_xyxy(t.box.tolist(), d["box"])
                if i > best_iou:
                    best_iou = i
                    best_tid = tid

            if best_tid is not None and best_iou >= TRACK_IOU_THRES:
                t = self.tracks[best_tid]
                t.update(d["box"], d["conf"], frame_id, timestamp)
                assigned_tracks.add(best_tid)

                d["track_id"] = int(t.id)
                d["track_confirmed"] = bool(t.confirmed)
                vx, vy, speed = t.get_velocity()
                d["velocity"] = {"vx": vx, "vy": vy, "speed": speed}
            else:
                tid = self.next_id
                self.next_id += 1
                t = Track(tid, d["box"], d["cls"], d["conf"], d.get("label", ""), frame_id, timestamp)
                self.tracks[tid] = t
                assigned_tracks.add(tid)

                d["track_id"] = int(tid)
                d["track_confirmed"] = bool(t.confirmed)
                d["velocity"] = {"vx": 0.0, "vy": 0.0, "speed": 0.0}

        # kill aged out tracks
        dead_ids = []
        for tid, t in self.tracks.items():
            if t.age > TRACK_MAX_AGE:
                t.kill(frame_id, timestamp)
                self.lifecycle_events.append(t.to_lifecycle_dict())
                dead_ids.append(tid)

        for tid in dead_ids:
            del self.tracks[tid]

        return det_list

    def flush_active(self, frame_id, timestamp):
        """
        Optional: call at end of run to record remaining active tracks as ended.
        """
        for tid, t in list(self.tracks.items()):
            t.kill(frame_id, timestamp)
            self.lifecycle_events.append(t.to_lifecycle_dict())
            del self.tracks[tid]

    def get_lifecycle_summary(self):
        return {
            "total_tracks_created": int(self.next_id - 1),
            "active_tracks": int(len(self.tracks)),
            "dead_tracks": int(len(self.lifecycle_events)),
            "lifecycle_events": self.lifecycle_events,
        }
