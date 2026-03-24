import cv2
import time
from pathlib import Path
import numpy as np
import onnxruntime as ort

def preprocess(img, size=640):
    img = cv2.resize(img, (size, size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def main():
    run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path("models") / "yolov8n.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not detected (try changing VideoCapture(0) to 1).")

    frame_id = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = preprocess(frame)
        t_infer = time.time()
        _ = session.run(None, {input_name: inp})
        infer_ms = (time.time() - t_infer) * 1000.0

        frame_id += 1
        fps = frame_id / max(1e-6, (time.time() - t0))

        cv2.putText(frame, f"FPS {fps:.1f} | infer {infer_ms:.1f}ms",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("MAPLE SHIELD (Windows)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("OK: pipeline ran")

if __name__ == "__main__":
    main()

