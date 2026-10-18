from pathlib import Path

import numpy as np

from ml.preprocessing import flat_vector, to_gray


class BaselineClassifier:
    """LDA on flattened pixels, saved by training/train_baseline.py."""

    name = "baseline"

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Baseline model not found: {path}")
        import joblib

        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.classes = bundle["classes"]
        self.size = bundle["img"]

    def predict(self, bgr, topk: int = 3):
        x = flat_vector(to_gray(bgr), self.size)[None]
        proba = self.model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1][:topk]
        return [
            {"label": self.classes[int(self.model.classes_[i])],
             "confidence": float(proba[i]), "box": None}
            for i in order
        ]
