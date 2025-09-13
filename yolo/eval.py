import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model on gestures dataset")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights (e.g., best.pt)")
    parser.add_argument("--data", type=str, default="yolo/gestures.yaml", help="Dataset yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    # Metrics include precision, recall, mAP50, mAP50-95 per class and overall
    print(metrics)


if __name__ == "__main__":
    main()


