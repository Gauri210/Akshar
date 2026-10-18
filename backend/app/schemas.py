from typing import Optional

from pydantic import BaseModel


class Detection(BaseModel):
    label: str
    confidence: float
    # [x1, y1, x2, y2] in pixels of the submitted image; None for classifiers
    box: Optional[list[float]] = None


class PredictResponse(BaseModel):
    model: str
    text: str
    detections: list[Detection]
    latency_ms: float
    image_size: list[int]  # [width, height]
