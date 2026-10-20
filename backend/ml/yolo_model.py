from pathlib import Path

import cv2
import numpy as np

from ml.preprocessing import crop_square, to_gray


class YoloModel:
    """Ultralytics YOLO. Classification weights (current scope: one character at a time) return the
    top-k classes; detection weights (future multi-character work) return boxes sorted left to right."""

    name = "yolo"

    def __init__(self, weights, conf: float = 0.25, imgsz: int = 640):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        from ultralytics import YOLO  # lazy: keeps the API importable without torch

        self.model = YOLO(str(weights))
        self.classify = self.model.task == "classify"
        self.conf = conf
        # classification: use the size it was trained at, so inputs match training
        self.imgsz = int(self.model.overrides.get("imgsz") or 64) if self.classify else imgsz

    def predict(self, bgr, topk: int = 3):
        return self._classify(bgr, topk) if self.classify else self._detect(bgr)

    def _classify(self, bgr, topk):
        # same crop as training and the CNN: tight crop around the ink, square, resized
        sq = crop_square(to_gray(bgr), self.imgsz)
        img = cv2.cvtColor(sq, cv2.COLOR_GRAY2BGR)
        res = self.model.predict(img, imgsz=self.imgsz, verbose=False)[0]
        probs = res.probs.data.cpu().numpy()
        order = np.argsort(probs)[::-1][:topk]
        return [{"label": res.names[int(i)], "confidence": float(probs[i]), "box": None}
                for i in order]

    def _detect(self, bgr):
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({"label": res.names[int(b.cls[0])], "confidence": float(b.conf[0]),
                         "box": [x1, y1, x2, y2]})
        dets.sort(key=lambda d: (d["box"][0] + d["box"][2]) / 2)
        return dets
