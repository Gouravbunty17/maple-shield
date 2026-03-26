import time
import cv2
import numpy as np
import onnxruntime as ort

# COCO80 labels (standard)
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def preprocess_bgr(frame, size=640):
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)   # NCHW
    return img

def xywh_to_xyxy(xywh):
    # xywh: (N,4) in 640-space
    x, y, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)

def nms_xyxy(boxes, scores, iou_thresh=0.45):
    # boxes: (N,4) xyxy, scores: (N,)
    if boxes.size == 0:
        return []

    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1e-6) * (y2 - y1 + 1e-6)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep

def postprocess_yolov8(out, conf_thresh=0.35, iou_thresh=0.45, max_det=50):
    """
    Ultralytics YOLOv8 ONNX export produced: (1, 84, 8400)
    Meaning: for each of 8400 candidates -> 4 box + 80 class scores
    There is no separate objectness; confidence = max(class_scores).
    """
    # out: (1,84,8400) -> (8400,84)
    pred = out[0].transpose(1, 0).astype(np.float32)

    boxes_xywh = pred[:, 0:4]
    cls_scores = pred[:, 4:]  # (8400,80)

    cls_id = np.argmax(cls_scores, axis=1)
    conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]

    mask = conf >= conf_thresh
    boxes_xywh = boxes_xywh[mask]
    conf = conf[mask]
    cls_id = cls_id[mask]

    if conf.size == 0:
        return np.zeros((0,6), dtype=np.float32)

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    # NMS (class-agnostic for V1 simplicity)
    keep = nms_xyxy(boxes_xyxy, conf, iou_thresh=iou_thresh)
    keep = keep[:max_det]

    det = np.stack([
        boxes_xyxy[keep, 0],
        boxes_xyxy[keep, 1],
        boxes_xyxy[keep, 2],
        boxes_xyxy[keep, 3],
        conf[keep],
        cls_id[keep].astype(np.float32)
    ], axis=1)

    return det  # (N,6): x1,y1,x2,y2,conf,cls

def scale_boxes(det, imgsz, frame_w, frame_h):
    if det.shape[0] == 0:
        return det
    sx = frame_w / float(imgsz)
    sy = frame_h / float(imgsz)
    det2 = det.copy()
    det2[:, 0] *= sx
    det2[:, 1] *= sy
    det2[:, 2] *= sx
    det2[:, 3] *= sy
    # clip
    det2[:, 0] = np.clip(det2[:, 0], 0, frame_w - 1)
    det2[:, 1] = np.clip(det2[:, 1], 0, frame_h - 1)
    det2[:, 2] = np.clip(det2[:, 2], 0, frame_w - 1)
    det2[:, 3] = np.clip(det2[:, 3], 0, frame_h - 1)
    return det2

def main():
    model_path = r"models\yolov8n.onnx"
    imgsz = 640

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not detected. Try VideoCapture(1) if needed.")

    frame_id = 0
    t0 = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        inp = preprocess_bgr(frame, imgsz)

        t_infer0 = time.time()
        out = sess.run(None, {input_name: inp})[0]  # (1,84,8400)
        infer_ms = (time.time() - t_infer0) * 1000.0

        det = postprocess_yolov8(out, conf_thresh=0.35, iou_thresh=0.45, max_det=50)
        det = scale_boxes(det, imgsz, w, h)

        # draw
        for x1, y1, x2, y2, conf, cls in det:
            x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
            cls = int(cls)
            name = COCO80[cls] if 0 <= cls < len(COCO80) else f"cls{cls}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_id += 1
        fps = frame_id / max(1e-6, (time.time() - t0))
        cv2.putText(frame, f"FPS {fps:.1f} | infer {infer_ms:.1f}ms | det {det.shape[0]}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("MAPLE SHIELD - Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

