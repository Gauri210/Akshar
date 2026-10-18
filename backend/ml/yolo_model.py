from pathlib import Path


class YoloDetector:
    name = "yolo"

    def __init__(self, weights, conf: float = 0.25, imgsz: int = 640):
        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        from ultralytics import YOLO  # lazy: keeps the API importable without torch

        self.model = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def predict(self, bgr):
        res = self.model.predict(bgr, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        dets = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            dets.append({
                "label": res.names[int(b.cls[0])],
                "confidence": float(b.conf[0]),
                "box": [x1, y1, x2, y2],
            })
        dets.sort(key=lambda d: (d["box"][0] + d["box"][2]) / 2)  # left to right
        return dets
