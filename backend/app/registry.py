import logging

from app import config

log = logging.getLogger("registry")


def _make_yolo():
    from ml.yolo_model import YoloDetector
    return YoloDetector(config.MODELS_DIR / config.YOLO_FILE, config.CONF_THRESHOLD, config.YOLO_IMGSZ)


def _make_baseline():
    from ml.baseline_model import BaselineClassifier
    return BaselineClassifier(config.MODELS_DIR / config.BASELINE_FILE)


def _make_cnn():
    from ml.cnn_model import CnnClassifier
    return CnnClassifier(config.MODELS_DIR / config.CNN_FILE)


FACTORIES = {"yolo": _make_yolo, "baseline": _make_baseline, "cnn": _make_cnn}


class ModelRegistry:
    def __init__(self):
        self.models = {}
        self.errors = {}

    def load_all(self):
        for name, factory in FACTORIES.items():
            try:
                self.models[name] = factory()
                log.info("loaded model %s", name)
            except Exception as e:  # a missing model must not stop the API
                self.errors[name] = f"{type(e).__name__}: {e}"
                log.warning("model %s unavailable: %s", name, e)

    def get(self, name: str):
        if name not in FACTORIES:
            raise KeyError(name)
        if name not in self.models:
            raise LookupError(self.errors.get(name, "model not loaded"))
        return self.models[name]
