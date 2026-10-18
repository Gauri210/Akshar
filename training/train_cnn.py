"""CNN for MODI character classification with a linear-SVM output layer.

--head svm      final Linear layer trained with squared hinge loss (L2-SVM, multi-class), as in
                "Deep Learning using Linear Support Vector Machines" (Tang, 2013). --wd supplies
                the margin regulariser.
--head softmax  same network trained with cross-entropy (use for an ablation).

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import load_crops, select_and_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="root folder with one subfolder per class")
    p.add_argument("--out", default=None, help="default: models/cnn_<head>.pth")
    p.add_argument("--report-dir", default=None, help="default: reports/cnn_<head>")
    p.add_argument("--per-class", type=int, default=300,
                   help="images kept per class (0 = all); use the same value for every script")
    p.add_argument("--img", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--head", choices=["svm", "softmax"], default="svm",
                   help="svm: linear SVM output layer (squared hinge loss); softmax: cross-entropy")
    p.add_argument("--wd", type=float, default=1e-4,
                   help="weight decay (acts as the SVM margin regulariser)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    args.out = args.out or f"models/cnn_{args.head}.pth"
    args.report_dir = args.report_dir or f"reports/cnn_{args.head}"
    return args


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
            nn.Linear(128, num_classes),  # the "SVM layer" when --head svm
        )

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

    sizes = tuple(len(split[k]) for k in ("train", "val", "test"))
    return mk("train", True, True), mk("val", False, False), mk("test", False, False), classes, sizes


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
def time_inference(model, device, img, n=200):
    model.eval()
    x = torch.randn(1, 1, img, img, device=device)
    for _ in range(20):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, test_dl, classes, sizes = build_data(args)
    print(f"device={device} head={args.head} classes={len(classes)} train/val/test={sizes}")

    model = ModiCNN(len(classes), args.img).to(device)
    if args.head == "svm":
        criterion = nn.MultiMarginLoss(p=2, margin=1.0)  # squared hinge = L2-SVM
    else:
        criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    history, best_val = [], 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        sched.step()
        val_acc, val_loss, _, _ = evaluate(model, val_dl, device, criterion)
        history.append(dict(epoch=epoch, train_loss=run_loss / n, val_loss=val_loss, val_acc=float(val_acc)))
        print(f"epoch {epoch:02d} train_loss={run_loss / n:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save({"state_dict": model.state_dict(), "classes": classes, "img": args.img,
                        "head": args.head}, args.out)

    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    test_acc, _, preds, labels = evaluate(model, test_dl, device)
    ms = time_inference(model, device, args.img)
    print(f"\nTEST accuracy={test_acc:.4f} | inference={ms:.2f} ms/image on {device.type}")

    names = list(range(len(classes)))
    report = classification_report(labels, preds, labels=names, target_names=classes,
                                   digits=3, zero_division=0)
    (report_dir / "classification_report.txt").write_text(report)
    print(report)

    cm = confusion_matrix(labels, preds, labels=names)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.title(f"Confusion matrix (test) - CNN, {args.head} head")
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
        "head": args.head, "weight_decay": args.wd,
        "test_accuracy": float(test_acc), "best_val_accuracy": float(best_val),
        "inference_ms_per_image": ms, "device": device.type,
        "num_classes": len(classes), "split_sizes": sizes, "history": history,
    }, indent=2))
    print(f"saved: {args.out} and reports in {report_dir}/")


if __name__ == "__main__":
    main()
