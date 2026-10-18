"""Image decoding and preprocessing shared by the API and the training scripts."""
import base64

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def decode_data_url(s: str) -> np.ndarray:
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    return decode_image(base64.b64decode(s))


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def normalize_polarity(gray: np.ndarray) -> np.ndarray:
    """Make ink dark on a light background."""
    return 255 - gray if gray.mean() < 127 else gray


def crop_square(gray: np.ndarray, size: int, pad: float = 0.15) -> np.ndarray:
    """Tight-crop the ink, pad to a square on white, resize to size x size.

    Intended for single, isolated characters. Full scenes go through YOLO instead.
    """
    gray = normalize_polarity(gray)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(mask > 0)
    crop = gray if len(xs) == 0 else gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(1, int(max(h, w) * (1 + 2 * pad)))
    canvas = np.full((side, side), 255, np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = crop
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def flat_vector(gray: np.ndarray, size: int = 32) -> np.ndarray:
    """Flat pixel vector for the LDA baseline (ink = high values)."""
    sq = crop_square(gray, size)
    return (1.0 - sq.astype(np.float32) / 255.0).ravel()


def cnn_array(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """(1, 1, size, size) float32 array normalised to [-1, 1]."""
    sq = crop_square(gray, size).astype(np.float32) / 255.0
    return ((sq - 0.5) / 0.5)[None, None]
