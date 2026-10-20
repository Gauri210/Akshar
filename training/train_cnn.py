"""CNN + SVC for MODI character classification.

Stage 1: train a small CNN with cross-entropy on cropped characters.
Stage 2: take the 128-d vector from its penultimate layer as features, standardise them
         (scaler fitted on the training set only) and fit an sklearn SVC on them. The kernel and C
         are chosen on the validation set, then the SVC is wrapped in a sigmoid calibrator so the
         API can report probabilities.

Both stages use the shared split from common.py. The report lists CNN-alone and CNN+SVC accuracy.

Expected data layout (one folder per class):
    DATA_ROOT/class_a/*.png, DATA_ROOT/class_b/*.png, ...

Usage:
    python training/train_cnn.py --data data/processed/offline_subset --per-class 300
"""
import argparse
import json
import random
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import load_crops, select_and_split

# (kernel, C) candidates for the SVC, compared on the validation set
SVC_CANDIDATES = [("linear", 1), ("rbf", 1), ("rbf", 10), ("rbf", 100)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="root folder with one subfolder per class")
    p.add_argument("--out", default="models/cnn.pth", help="CNN weights")
    p.add_argument("--svc-out", default="models/svc.joblib", help="scaler + SVC")
    p.add_argument("--report-dir", default="reports/cnn_svc")
    p.add_argument("--per-class", type=int, default=300,
                   help="images kept per class (0 = all); use the same value for every script")
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModiCNN(nn.Module):
    """Conv(32) -> Conv(64) -> Dense(128) -> Dense(num_classes). Keep in sync with backend/ml/cnn_model.py."""

    def __init__(self, num_classes, img=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        flat = 64 * (img // 4) * (img // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def embed(self, x):
        """128-d feature vector: everything before the last Linear layer. This is the SVC's input."""
        return self.classifier[:-1](self.features(x))

    def forward(self, x):
        return self.classifier(self.features(x))


class CropDataset(Dataset):
    """Cropped characters (uint8, dark ink on white), same preprocessing as the API."""

    def __init__(self, crops, labels, augment):
        self.x = torch.from_numpy(crops)
        self.y = torch.from_numpy(labels).long()
        self.aug = (transforms.RandomAffine(10, translate=(0.08, 0.08), scale=(0.9, 1.1), fill=1.0)
                    if augment else None)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i].float().div(255.0).unsqueeze(0)  # (1, H, W) in [0, 1], background = 1
        if self.aug is not None:
            x = self.aug(x)
        return (x - 0.5) / 0.5, self.y[i]


def build_data(args):
    classes, paths, labels, split = select_and_split(args.data, args.per_class, args.seed)
    crops = load_crops(paths, args.img)

    def mk(name, augment, shuffle):
        ids = split[name]
        return DataLoader(CropDataset(crops[ids], labels[ids], augment), batch_size=args.bs,
                          shuffle=shuffle, num_workers=args.workers)

    loaders = {
        "train": mk("train", True, True),           # augmented, for training the CNN
        "train_eval": mk("train", False, False),    # same images, no augmentation, for features
        "val": mk("val", False, False),
        "test": mk("test", False, False),
    }
    y = {k: labels[split[k]] for k in ("train", "val", "test")}
    sizes = tuple(len(split[k]) for k in ("train", "val", "test"))
    return loaders, y, classes, sizes


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    preds, labels, loss_sum = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * len(y)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return (preds == labels).mean(), loss_sum / len(labels), preds, labels


@torch.no_grad()
def embed(model, loader, device):
    model.eval()
    return torch.cat([model.embed(x.to(device)).cpu() for x, _ in loader]).numpy()


def fit_svc(Ztr, ytr, Zva, yva):
    """Pick (kernel, C) on the validation set; return (pipeline, candidate results, best config)."""
    scaler = StandardScaler().fit(Ztr)  # statistics from the training set only
    Str, Sva = scaler.transform(Ztr), scaler.transform(Zva)
    results, best_cfg, best_val = [], None, -1.0
    for kernel, C in SVC_CANDIDATES:
        svc = SVC(kernel=kernel, C=C, gamma="scale").fit(Str, ytr)
        tr = accuracy_score(ytr, svc.predict(Str))
        va = accuracy_score(yva, svc.predict(Sva))
        results.append({"kernel": kernel, "C": C, "train_acc": float(tr), "val_acc": float(va)})
        print(f"SVC kernel={kernel:6s} C={C:<4} train={tr:.4f} val={va:.4f}")
        if va > best_val:
            best_cfg, best_val = (kernel, C), va
    # Sigmoid calibration gives predict_proba for the API (SVC(probability=True) is deprecated)
    final = CalibratedClassifierCV(SVC(kernel=best_cfg[0], C=best_cfg[1], gamma="scale"),
                                   method="sigmoid", cv=3, ensemble=False).fit(Str, ytr)
    return Pipeline([("scale", scaler), ("svc", final)]), results, best_cfg


@torch.no_grad()
def time_chain(model, pipe, device, img, n=100):
    """Milliseconds per image for the whole CNN -> SVC chain (as the API runs it)."""
    model.eval()
    x = torch.randn(1, 1, img, img, device=device)
    for _ in range(10):
        pipe.predict_proba(model.embed(x).cpu().numpy())
    t = time.perf_counter()
    for _ in range(n):
        pipe.predict_proba(model.embed(x).cpu().numpy())
    return (time.perf_counter() - t) / n * 1000


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.svc_out).parent.mkdir(parents=True, exist_ok=True)

    loaders, y, classes, sizes = build_data(args)
    print(f"device={device} classes={len(classes)} train/val/test={sizes}")

    model = ModiCNN(len(classes), args.img).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    # ---- stage 1: CNN ----
    history, best_val = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, yb in loaders["train"]:
            x, yb = x.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(x), yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        val_acc, val_loss, _, _ = evaluate(model, loaders["val"], device, criterion)
        history.append(dict(epoch=epoch, train_loss=run_loss / n, val_loss=val_loss, val_acc=float(val_acc)))
        print(f"epoch {epoch:02d} train_loss={run_loss / n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "img": args.img}, args.out)

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    cnn_test_acc, _, _, _ = evaluate(model, loaders["test"], device)
    print(f"\nCNN alone: TEST accuracy={cnn_test_acc:.4f}")

    # ---- stage 2: SVC on the CNN's 128-d features ----
    Z = {k: embed(model, loaders[k], device) for k in ("train_eval", "val", "test")}
    pipe, svc_results, best_cfg = fit_svc(Z["train_eval"], y["train"], Z["val"], y["val"])
    preds = pipe.predict(Z["test"])
    svc_test_acc = float(accuracy_score(y["test"], preds))
    ms = time_chain(model, pipe, device, args.img)
    print(f"CNN + SVC {best_cfg}: TEST accuracy={svc_test_acc:.4f} | {ms:.2f} ms/image on {device.type}")
    joblib.dump({"pipeline": pipe, "classes": classes, "img": args.img, "name": "cnn_svc"}, args.svc_out)

    names = list(range(len(classes)))
    report = classification_report(y["test"], preds, labels=names, target_names=classes,
                                   digits=3, zero_division=0)
    (report_dir / "classification_report.txt").write_text(report)
    print(report)

    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix(y["test"], preds, labels=names), cmap="Blues")
    plt.colorbar()
    plt.title("Confusion matrix (test) - CNN + SVC")
    plt.xlabel("Predicted class index")
    plt.ylabel("True class index")
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    plt.plot(epochs, [h["val_loss"] for h in history], label="val loss")
    plt.plot(epochs, [h["val_acc"] for h in history], label="val acc")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "training_curves.png", dpi=150)
    plt.close()

    (report_dir / "metrics.json").write_text(json.dumps({
        "cnn_test_accuracy": float(cnn_test_acc), "cnn_best_val_accuracy": float(best_val),
        "cnn_svc_test_accuracy": svc_test_acc, "svc_best": {"kernel": best_cfg[0], "C": best_cfg[1]},
        "svc_candidates": svc_results, "inference_ms_per_image": ms, "device": device.type,
        "num_classes": len(classes), "split_sizes": sizes, "history": history,
    }, indent=2))
    print(f"saved: {args.out}, {args.svc_out} and reports in {report_dir}/")


if __name__ == "__main__":
    main()
