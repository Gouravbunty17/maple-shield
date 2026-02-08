import json
import sys
from pathlib import Path
import cv2

def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def main():
    if len(sys.argv) < 2:
        print("Usage: python borealis_v1_replay.py runs\\<timestamp>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    video_path = run_dir / "overlay.mp4"
    jsonl_path = run_dir / "detections.jsonl"

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing jsonl: {jsonl_path}")

    cap = cv2.VideoCapture(str(video_path))
    logs = read_jsonl(jsonl_path)

    print(f"Replaying: {run_dir}")
    print("Keys: space=pause/resume | q=quit")

    paused = False
    next_log = next(logs, None)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            # Show last log record info on screen
            if next_log is not None:
                ts = next_log.get("ts")
                fr = next_log.get("frame")
                infer = next_log.get("infer_ms")
                detn = len(next_log.get("detections", []))
                cv2.putText(frame, f"LOG frame={fr} ts={ts:.3f} infer={infer:.1f} det={detn}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                next_log = next(logs, None)

            cv2.imshow("BOREALIS Replay", frame)

        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
