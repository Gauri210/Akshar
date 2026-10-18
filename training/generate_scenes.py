"""Generate synthetic multi-character scenes with YOLO labels from isolated character images.

Source images are split into train/val/test BEFORE pasting, so no source character
image appears in more than one split (no leakage). The split comes from common.select_and_split,
the same function the LDA and CNN scripts use, so all three methods are tested on scenes and
crops built from the same held-out source images. Pass the same --per-class and --seed to all.

Usage:
    python training/generate_scenes.py --src data/processed/offline_subset \
        --out data/processed/yolo_scenes --n-train 4000 --n-val 500 --n-test 500
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from common import select_and_split


def load_char(path, side, rng):
    """Return an ink-strength mask (0-255), tight-cropped, rotated slightly."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    if g.mean() < 127:
        g = 255 - g
    mask = 255 - g
    ys, xs = np.where(mask > 40)
    if len(xs) == 0:
        return None
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h0, w0 = mask.shape
    scale = side / max(h0, w0)
    w, h = max(8, int(w0 * scale)), max(8, int(h0 * scale))
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)

    ang = rng.uniform(-8, 8)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    mask = cv2.warpAffine(mask, M, (nw, nh))
    ys, xs = np.where(mask > 40)
    if len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def make_scene(pool, size, rng, nrng, nmin, nmax):
    chars, used = [], 0
    for _ in range(rng.randint(nmin, nmax)):
        cls, path = rng.choice(pool)
        m = load_char(path, rng.randint(48, 110), rng)
        if m is None:
            continue
        gap = rng.randint(8, 40)
        if used + m.shape[1] + gap > size - 40:
            break
        chars.append((cls, m, gap))
        used += m.shape[1] + gap
    if not chars:
        return None

    canvas = np.full((size, size), rng.randint(190, 255), np.float32)
    canvas += nrng.normal(0, rng.uniform(2, 10), canvas.shape)
    canvas *= np.linspace(rng.uniform(0.85, 1.0), 1.0, size, dtype=np.float32)[None, :]

    ink = rng.randint(0, 70)
    x = rng.randint(10, max(10, size - 10 - used))
    base_y = rng.randint(60, size - 60)
    labels = []
    for cls, m, gap in chars:
        h, w = m.shape
        if x + w > size - 5:
            break
        y = int(base_y - h / 2 + rng.randint(-10, 10))
        y = min(max(y, 5), size - h - 5)
        a = m.astype(np.float32) / 255.0
        region = canvas[y:y + h, x:x + w]
        canvas[y:y + h, x:x + w] = region * (1 - a) + ink * a
        labels.append((cls, (x + w / 2) / size, (y + h / 2) / size, w / size, h / size))
        x += w + gap
    if not labels:
        return None

    k = rng.choice([1, 3, 5])
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img, labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder-per-class isolated characters")
    p.add_argument("--out", default="data/processed/yolo_scenes")
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--n-test", type=int, default=500)
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--min-chars", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=6)
    p.add_argument("--per-class", type=int, default=300, help="use the same value for every script")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    out = Path(args.out).resolve()
    classes, paths, labels, split_idx = select_and_split(args.src, args.per_class, args.seed)
    pools = {s: [(int(labels[i]), paths[i]) for i in idx] for s, idx in split_idx.items()}

    counts = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    for split, n in counts.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        if not pools[split]:
            raise SystemExit(f"no source images for split '{split}'")
        made = 0
        while made < n:
            res = make_scene(pools[split], args.img_size, rng, nrng, args.min_chars, args.max_chars)
            if res is None:
                continue
            img, labels = res
            stem = f"{split}_{made:06d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with open(out / "labels" / split / f"{stem}.txt", "w") as f:
                for cls, cx, cy, w, h in labels:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            made += 1
        print(f"{split}: {made} scenes")

    yaml = [f"path: {out}", "train: images/train", "val: images/val", "test: images/test", "names:"]
    yaml += [f"  {i}: {json.dumps(c)}" for i, c in enumerate(classes)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"wrote {out / 'data.yaml'} ({len(classes)} classes)")


if __name__ == "__main__":
    main()
