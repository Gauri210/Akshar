"""Fine-tune a YOLO classification model on the same split as the CNN + SVC.

The shared split from common.py is written out as train/val/test folders of cropped characters
(the same crops the API produces), then Ultralytics trains on them. Horizontal and vertical flips
are turned off because a mirrored MODI character is a different (or invalid) character.

Usage:
    python training/train_yolo_cls.py --data data/processed/offline_subset
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import cv2

from common import load_crops, select_and_split


def materialize(args):
    """Write the shared split as <work-dir>/{train,val,test}/<class>/*.png."""
    classes, paths, labels, split = select_and_split(args.data, args.per_class, args.seed)
    crops = load_crops(paths, args.imgsz)
    root = Path(args.work_dir)
    if root.exists():
        shutil.rmtree(root)
    for name, ids in split.items():
        for n, i in enumerate(ids):
            d = root / name / classes[labels[i]]
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{n:05d}.png"), crops[i])
    sizes = {k: len(v) for k, v in split.items()}
    return root, classes, sizes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder-per-class dataset (same as train_cnn.py)")
    p.add_argument("--per-class", type=int, default=300, help="use the same value for every script")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", default="yolov8n-cls.pt", help="e.g. yolov8n-cls.pt or yolo11n-cls.pt")
    p.add_argument("--imgsz", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None, help="e.g. 0 for GPU, cpu")
    p.add_argument("--work-dir", default="data/processed/yolo_cls")
    p.add_argument("--project", default="runs/yolo_cls")
    p.add_argument("--name", default="modi")
    p.add_argument("--out", default="models/yolo_cls.pt")
    p.add_argument("--report-dir", default="reports/yolo_cls")
    args = p.parse_args()

    root, classes, sizes = materialize(args)
    print(f"classes={len(classes)} split sizes={sizes} -> {root}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(data=str(root.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, device=args.device, project=args.project, name=args.name,
                exist_ok=True, patience=10, seed=args.seed, plots=False,
                fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, args.out)

    final = YOLO(args.out)
    m = final.val(data=str(root.resolve()), split="test", imgsz=args.imgsz, device=args.device,
                  plots=False)

    sample = cv2.imread(str(next((root / "test").rglob("*.png"))))
    for _ in range(5):
        final.predict(sample, imgsz=args.imgsz, verbose=False)
    t = time.perf_counter()
    for _ in range(50):
        final.predict(sample, imgsz=args.imgsz, verbose=False)
    ms = (time.perf_counter() - t) / 50 * 1000

    metrics = {"top1_accuracy": float(m.top1), "top5_accuracy": float(m.top5),
               "inference_ms_per_image": ms, "num_classes": len(classes), "split_sizes": sizes,
               "weights": args.weights, "imgsz": args.imgsz}
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("TEST metrics:", metrics)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
