import time

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app import config
from app.schemas import Detection, PredictResponse
from ml.preprocessing import decode_data_url, decode_image

router = APIRouter()


def build_text(dets: list[dict]) -> str:
    boxed = [d for d in dets if d.get("box")]
    if boxed:  # detector output is already sorted left to right
        return " ".join(d["label"] for d in boxed)
    return dets[0]["label"] if dets else ""


def run_model(registry, name: str, img) -> PredictResponse:
    model = registry.get(name)
    t = time.perf_counter()
    dets = model.predict(img)
    ms = (time.perf_counter() - t) * 1000
    h, w = img.shape[:2]
    return PredictResponse(
        model=name,
        text=build_text(dets),
        detections=[Detection(**d) for d in dets],
        latency_ms=round(ms, 2),
        image_size=[w, h],
    )


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/models")
def models(request: Request):
    reg = request.app.state.registry
    return {"available": sorted(reg.models), "unavailable": reg.errors}


@router.post("/predict/{name}", response_model=PredictResponse)
async def predict(name: str, request: Request, file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "file too large")
    try:
        img = decode_image(data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        return await run_in_threadpool(run_model, request.app.state.registry, name, img)
    except KeyError:
        raise HTTPException(404, f"unknown model '{name}'")
    except LookupError as e:
        raise HTTPException(503, f"model '{name}' unavailable: {e}")


@router.websocket("/ws/{name}")
async def stream(ws: WebSocket, name: str):
    """Client sends JPEG frames (data URL text or raw bytes); server replies with JSON."""
    await ws.accept()
    registry = ws.app.state.registry
    try:
        registry.get(name)
    except (KeyError, LookupError) as e:
        await ws.send_json({"error": f"model '{name}' unavailable: {e}"})
        await ws.close(code=1008)
        return
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            try:
                img = decode_data_url(msg["text"]) if msg.get("text") else decode_image(msg["bytes"])
                resp = await run_in_threadpool(run_model, registry, name, img)
                await ws.send_text(resp.model_dump_json())
            except ValueError as e:
                await ws.send_json({"error": str(e)})
    except WebSocketDisconnect:
        pass
