import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on hand gesture dataset")
    parser.add_argument("--data", type=str, default="yolo/gestures.yaml", help="Path to dataset yaml")
    parser.add_argument("--model", type=str, default="yolov8n.yaml", help="Base model config or weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--project", type=str, default="runs/gestures", help="Project output dir")
    parser.add_argument("--name", type=str, default="yolov8n-gestures", help="Run name")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    assert data_path.exists(), f"Dataset yaml not found: {data_path}"

    model = YOLO(args.model)

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        pretrained=True,
    )

    # Save best weights path
    print("Training complete. Best weights:", model.ckpt_path)


if __name__ == "__main__":
    main()


