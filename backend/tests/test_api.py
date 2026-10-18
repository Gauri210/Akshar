import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app


def png_bytes():
    img = np.full((64, 64, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 10), (40, 55), (0, 0, 0), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"


def test_models_lists_all():
    with TestClient(app) as c:
        j = c.get("/models").json()
        assert set(j["available"]) | set(j["unavailable"]) == {"yolo", "baseline", "cnn"}


def test_unknown_model_404():
    with TestClient(app) as c:
        r = c.post("/predict/nope", files={"file": ("x.png", png_bytes(), "image/png")})
        assert r.status_code == 404


def test_unloaded_model_503():
    with TestClient(app) as c:
        j = c.get("/models").json()
        if "yolo" in j["unavailable"]:
            r = c.post("/predict/yolo", files={"file": ("x.png", png_bytes(), "image/png")})
            assert r.status_code == 503


def test_bad_image_400():
    with TestClient(app) as c:
        r = c.post("/predict/yolo", files={"file": ("x.png", b"not an image", "image/png")})
        assert r.status_code == 400
