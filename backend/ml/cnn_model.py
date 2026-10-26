from pathlib import Path

import numpy as np

from ml.preprocessing import cnn_array, to_gray


def build_model(num_classes: int, img: int = 64):
    """Keep in sync with ModiCNN in notebooks/02_cnn_svc.ipynb."""
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

        def embed(self, x):
            return self.classifier[:-1](self.features(x))

        def forward(self, x):
            return self.classifier(self.features(x))

    return ModiCNN()


class CnnSvcClassifier:
    """CNN feature extractor (128-d) followed by a scaler + calibrated SVC."""

    name = "cnn_svc"

    def __init__(self, cnn_path, svc_path):
        cnn_path, svc_path = Path(cnn_path), Path(svc_path)
        for p in (cnn_path, svc_path):
            if not p.exists():
                raise FileNotFoundError(f"model file not found: {p}")
        import joblib
        import torch

        self.torch = torch
        ckpt = torch.load(cnn_path, map_location="cpu")
        bundle = joblib.load(svc_path)
        if list(ckpt["classes"]) != list(bundle["classes"]):
            raise ValueError("CNN weights and SVC were trained on different classes")
        self.classes = ckpt["classes"]
        self.size = ckpt["img"]
        self.svc = bundle["pipeline"]
        self.model = build_model(len(self.classes), self.size)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict(self, bgr, topk: int = 3):
        torch = self.torch
        x = torch.from_numpy(cnn_array(to_gray(bgr), self.size))
        with torch.no_grad():
            z = self.model.embed(x).numpy()
        proba = self.svc.predict_proba(z)[0]
        order = np.argsort(proba)[::-1][:topk]
        return [
            {"label": self.classes[int(self.svc.classes_[i])],
             "confidence": float(proba[i]), "box": None}
            for i in order
        ]
