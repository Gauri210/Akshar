import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
YOLO_FILE = os.getenv("YOLO_FILE", "yolo_best.pt")
BASELINE_FILE = os.getenv("BASELINE_FILE", "baseline_lda.joblib")
CNN_FILE = os.getenv("CNN_FILE", "cnn_svm.pth")

CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
