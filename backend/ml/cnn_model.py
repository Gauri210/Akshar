from pathlib import Path

from ml.preprocessing import cnn_array, to_gray


def build_model(num_classes: int, img: int = 64):
    """Keep in sync with ModiCNN in training/train_cnn.py."""
    import torch.nn as nn

    class ModiCNN(nn.Module):
        def __init__(self):
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

        def forward(self, x):
            return self.classifier(self.features(x))

    return ModiCNN()


class CnnClassifier:
    name = "cnn"

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CNN weights not found: {path}")
        import torch

        self.torch = torch
        ckpt = torch.load(path, map_location="cpu")
        self.classes = ckpt["classes"]
        self.size = ckpt["img"]
        self.head = ckpt.get("head", "softmax")
        self.model = build_model(len(self.classes), self.size)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, bgr, topk: int = 3):
        torch = self.torch
        x = torch.from_numpy(cnn_array(to_gray(bgr), self.size))
        with torch.no_grad():
            scores = self.model(x)[0]
        if self.head == "svm":
            # SVM decision scores are not probabilities. Map the margin range [-1, 1] to [0, 1]
            # for display only; the ranking and the predicted label come from the raw scores.
            _, idx = scores.topk(min(topk, len(self.classes)))
            conf = ((scores[idx] + 1) / 2).clamp(0, 1)
        else:
            conf, idx = torch.softmax(scores, dim=0).topk(min(topk, len(self.classes)))
        return [
            {"label": self.classes[int(i)], "confidence": float(c), "box": None}
            for c, i in zip(conf, idx)
        ]
