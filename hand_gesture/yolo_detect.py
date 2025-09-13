from typing import List, Tuple
import cv2

try:
    from ultralytics import YOLO  # type: ignore
except Exception as exc:
    raise ImportError("ultralytics is required. Install with `pip install ultralytics`.") from exc


class YoloGestureDetector:
    """YOLOv8 wrapper for gesture-as-class detection.

    Expects a detector trained on gesture classes like: thumbs_up, peace, fist, stop, okay.
    """

    def __init__(self, weights_path: str, imgsz: int = 640, conf: float = 0.35):
        self.model = YOLO(weights_path)
        self.imgsz = imgsz
        self.conf = conf

    def predict(self, frame_bgr) -> Tuple[any, List[Tuple[str, float]]]:
        """Run inference and return annotated frame and (label, confidence) list."""
        results = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        annotated = results[0].plot()
        detections: List[Tuple[str, float]] = []
        names = results[0].names
        boxes = results[0].boxes
        if boxes is not None:
            for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                label = names[int(cls_id)] if int(cls_id) in names else str(int(cls_id))
                detections.append((label, float(conf)))
        return annotated, detections


