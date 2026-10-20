"""Shared dataset selection, split and loading.

train_cnn.py (CNN + SVC) and train_yolo_cls.py (YOLO) both call select_and_split() with the same
folder, --per-class and --seed, so every source image lands in the same train/val/test partition
for both methods. Run them with identical values for those three arguments.
(generate_scenes.py, for future multi-character work, uses the same function.)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.ml.preprocessing import crop_square  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def select_and_split(root, per_class=300, seed=42):
    """Returns (classes, paths, labels, {"train": idx, "val": idx, "test": idx}); 70/15/15, stratified."""
    root = Path(root)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    if not classes:
        raise SystemExit(f"no class folders found in {root}")
    rng = np.random.RandomState(seed)
    paths, labels = [], []
    for ci, c in enumerate(classes):
        files = sorted(f for f in (root / c).iterdir() if f.suffix.lower() in EXTS)
        if per_class and len(files) > per_class:
            keep = np.sort(rng.choice(len(files), per_class, replace=False))
            files = [files[i] for i in keep]
        paths.extend(files)
        labels.extend([ci] * len(files))
    labels = np.array(labels)
    idx = np.arange(len(paths))
    tr, tmp = train_test_split(idx, test_size=0.3, stratify=labels, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=seed)
    return classes, paths, labels, {"train": tr, "val": va, "test": te}


def load_crops(paths, size):
    """(N, size, size) uint8: tight-cropped, padded square, dark ink on white (same as the API)."""
    out = np.full((len(paths), size, size), 255, np.uint8)
    bad = 0
    for i, p in enumerate(paths):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            bad += 1
            continue
        out[i] = crop_square(g, size)
    if bad:
        print(f"warning: {bad} unreadable images were left blank")
    return out
