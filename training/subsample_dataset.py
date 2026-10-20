"""Copy up to N random images per class into a flat, folder-per-class dataset.

Accepts both layouts:
    flat:    src/<class>/*.png
    grouped: src/<group>/<class>/*.png      (e.g. MODI-HChar: digits / consonants / vowels)

The output is always dst/<class>/*.png, which is what the training scripts expect. If two groups
contain a class with the same name, both are renamed to <group>_<class>.

Usage:
    python training/subsample_dataset.py --src data/raw/modi_hchar_full \
        --dst data/processed/offline_subset --per-class 300 [--max-classes 20]
"""
import argparse
import random
import shutil
from collections import Counter
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def has_images(d: Path) -> bool:
    return any(f.suffix.lower() in EXTS for f in d.iterdir() if f.is_file())


def discover(src: Path) -> dict:
    """Return {class_name: folder} for the flat or grouped layout."""
    found = []  # (group, class_name, folder)
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        if has_images(d):
            found.append(("", d.name, d))
        else:
            for sub in sorted(p for p in d.iterdir() if p.is_dir()):
                if has_images(sub):
                    found.append((d.name, sub.name, sub))
    counts = Counter(name for _, name, _ in found)
    return {(f"{g}_{n}" if g and counts[n] > 1 else n): folder for g, n, folder in found}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--per-class", type=int, default=300)
    p.add_argument("--max-classes", type=int, default=0, help="0 = all classes")
    p.add_argument("--classes", nargs="*", help="explicit class names to keep")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    folders = discover(src)
    if not folders:
        raise SystemExit(f"no folders containing images found under {src}")
    classes = sorted(folders)
    print(f"found {len(classes)} classes under {src}")
    if args.classes:
        classes = [c for c in classes if c in set(args.classes)]
    elif args.max_classes:
        classes = sorted(rng.sample(classes, min(args.max_classes, len(classes))))

    total = 0
    for c in classes:
        files = sorted(f for f in folders[c].iterdir() if f.suffix.lower() in EXTS)
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
