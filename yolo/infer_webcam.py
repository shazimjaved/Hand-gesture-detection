import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv8 webcam inference for gesture detection")
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights (best.pt)")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Gestures - Webcam", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


