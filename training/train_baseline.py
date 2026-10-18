"""Classical baseline: Linear Discriminant Analysis (Fisher LDA) on raw pixels. No PCA.

LDA is fitted on the flattened, cropped character image. Solver / shrinkage is chosen on the
validation set, since raw pixel covariance is singular or ill-conditioned (constant border pixels).

Usage:
    python training/train_baseline.py --data data/processed/offline_subset
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from common import load_crops, select_and_split

# (solver, shrinkage); shrinkage only applies to lsqr
CANDIDATES = [("svd", None), ("lsqr", "auto"), ("lsqr", 0.1), ("lsqr", 0.5)]


def features(crops):
    """Flat vector, ink = high. Must match backend.ml.preprocessing.flat_vector."""
    return 1.0 - crops.reshape(len(crops), -1).astype(np.float32) / 255.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="models/baseline_lda.joblib",
                   help="keep the default filename so the API (BASELINE_FILE) finds it")
    p.add_argument("--report-dir", default="reports/baseline")
    p.add_argument("--img", type=int, default=32)
    p.add_argument("--per-class", type=int, default=300, help="use the same value for every script")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    classes, paths, labels, split = select_and_split(args.data, args.per_class, args.seed)
    X = features(load_crops(paths, args.img))
    Xtr, ytr = X[split["train"]], labels[split["train"]]
    Xva, yva = X[split["val"]], labels[split["val"]]
    Xte, yte = X[split["test"]], labels[split["test"]]
    print(f"classes={len(classes)} features={X.shape[1]} "
          f"train/val/test={len(Xtr)}/{len(Xva)}/{len(Xte)}")

    results, best, best_val = [], None, -1.0
    for solver, shrink in CANDIDATES:
        lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink)
        lda.fit(Xtr, ytr)
        tr = accuracy_score(ytr, lda.predict(Xtr))
        va = accuracy_score(yva, lda.predict(Xva))
        results.append({"solver": solver, "shrinkage": shrink, "train_acc": float(tr), "val_acc": float(va)})
        print(f"LDA solver={solver:4s} shrinkage={str(shrink):5s} train={tr:.4f} val={va:.4f}")
        if va > best_val:
            best, best_val, best_cfg = lda, va, (solver, shrink)

    pred = best.predict(Xte)
    test_acc = float(accuracy_score(yte, pred))
    t = time.perf_counter()
    for _ in range(100):
        best.predict(Xte[:1])
    ms = (time.perf_counter() - t) / 100 * 1000
    print(f"best={best_cfg} val={best_val:.4f} TEST={test_acc:.4f} {ms:.2f} ms/img")

    joblib.dump({"model": best, "classes": classes, "img": args.img, "name": "lda"}, args.out)

    (report / "classification_report.txt").write_text(
        classification_report(yte, pred, labels=range(len(classes)), target_names=classes,
                              digits=3, zero_division=0))

    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix(yte, pred, labels=range(len(classes))), cmap="Blues")
    plt.colorbar()
    plt.title(f"Confusion matrix (test) - LDA {best_cfg}")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.tight_layout()
    plt.savefig(report / "confusion_matrix.png", dpi=150)
    plt.close()

    # First two Fisher discriminant axes (svd solver supports transform)
    if len(classes) > 2:
        proj = LinearDiscriminantAnalysis(solver="svd", n_components=2).fit(Xtr, ytr)
        Z = proj.transform(Xte)
        plt.figure(figsize=(7, 6))
        plt.scatter(Z[:, 0], Z[:, 1], c=yte, s=6, cmap="nipy_spectral")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        plt.title("Test set on the first two Fisher discriminants")
        plt.tight_layout()
        plt.savefig(report / "lda_projection.png", dpi=150)
        plt.close()

    (report / "metrics.json").write_text(json.dumps({
        "model": "lda", "solver": best_cfg[0], "shrinkage": best_cfg[1],
        "val_acc": float(best_val), "test_acc": test_acc, "ms_per_image": ms,
        "candidates": results,
        "split_sizes": [len(Xtr), len(Xva), len(Xte)],
    }, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
