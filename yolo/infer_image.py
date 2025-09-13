import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLOv8 inference on images for gesture detection")
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights (e.g., runs/.../best.pt)")
    p.add_argument("--source", type=str, required=True, help="Path to an image or a directory of images")
    p.add_argument("--imgsz", type=int, default=640, help="Image size")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    p.add_argument("--save", action="store_true", help="Save annotated outputs next to inputs")
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    src = Path(args.source)
    paths = [src] if src.is_file() else list(src.glob("*.jpg")) + list(src.glob("*.png"))
    assert paths, f"No images found at {src}"

    for p in paths:
        img = cv2.imread(str(p))
        results = model.predict(img, imgsz=args.imgsz, conf=args.conf, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("YOLOv8 Gestures", annotated)
        if args.save:
            out = p.with_name(p.stem + "_pred" + p.suffix)
            cv2.imwrite(str(out), annotated)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


