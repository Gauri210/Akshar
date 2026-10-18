"""Copy up to N random images per class into a smaller dataset.

Usage:
    python training/subsample_dataset.py --src data/raw/modi_hchar_full \
        --dst data/processed/offline_subset --per-class 300 [--max-classes 20]
"""
import argparse
import random
import shutil
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--max-classes", type=int, default=0, help="0 = all classes")
    p.add_argument("--classes", nargs="*", help="explicit class folder names to keep")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    classes = sorted(d.name for d in src.iterdir() if d.is_dir())
    if args.classes:
        classes = [c for c in classes if c in set(args.classes)]
    elif args.max_classes:
        classes = rng.sample(classes, min(args.max_classes, len(classes)))
        classes.sort()

    total = 0
    for c in classes:
        files = [f for f in (src / c).iterdir() if f.suffix.lower() in EXTS]
        rng.shuffle(files)
        files = files[: args.per_class] if args.per_class else files
        out = dst / c
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, out / f.name)
        total += len(files)
    print(f"copied {total} images across {len(classes)} classes to {dst}")


if __name__ == "__main__":
    main()
