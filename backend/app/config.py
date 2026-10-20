import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
YOLO_FILE = os.getenv("YOLO_FILE", "yolo_cls.pt")
CNN_FILE = os.getenv("CNN_FILE", "cnn.pth")
SVC_FILE = os.getenv("SVC_FILE", "svc.joblib")

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))  # detection weights only
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))  # detection weights only
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
