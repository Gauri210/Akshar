"""Fine-tune YOLO on the synthetic scene dataset and export the best weights.

Usage:
    python training/train_yolo.py --data data/processed/yolo_scenes/data.yaml
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/yolo_scenes/data.yaml")
    p.add_argument("--weights", default="yolov8n.pt", help="pretrained start, e.g. yolov8n.pt or yolo11n.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    p.add_argument("--project", default="runs/yolo")
    p.add_argument("--name", default="modi")
    p.add_argument("--out", default="models/yolo_best.pt")
    p.add_argument("--report-dir", default="reports/yolo")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project=args.project, name=args.name, exist_ok=True,
                patience=20, seed=args.seed)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)

    m = YOLO(args.out).val(data=args.data, split="test", imgsz=args.imgsz, device=args.device)
    metrics = {"map50": float(m.box.map50), "map50_95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr)}
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("test metrics:", metrics)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
