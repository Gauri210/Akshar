const PREDICT_DELAY_MS = 500; // wait this long after the pen lifts before predicting

const pad = document.getElementById("pad");
const ctx = pad.getContext("2d");
const modelSel = document.getElementById("model");
const penEl = document.getElementById("pen");
const clearBtn = document.getElementById("clear");
const textEl = document.getElementById("text");
const top3El = document.getElementById("top3");
const latencyEl = document.getElementById("latency");
const statusEl = document.getElementById("status");

let drawing = false, last = null, timer = null, hasInk = false, reqId = 0;

function clearPad() {
  clearTimeout(timer);
  reqId++; // any reply still in flight is now stale
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  hasInk = false;
  textEl.textContent = "-";
  top3El.textContent = "-";
  latencyEl.textContent = "-";
  statusEl.textContent = "draw a character";
}

function pos(e) {
  const r = pad.getBoundingClientRect();
  return { x: (e.clientX - r.left) * pad.width / r.width, y: (e.clientY - r.top) * pad.height / r.height };
}

function stroke(from, to) {
  ctx.strokeStyle = "#000";
  ctx.fillStyle = "#000";
  ctx.lineWidth = Number(penEl.value);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  hasInk = true;
}

pad.addEventListener("pointerdown", (e) => {
  pad.setPointerCapture(e.pointerId);
  clearTimeout(timer);
  drawing = true;
  last = pos(e);
  stroke(last, last); // a tap leaves a dot
});

pad.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const p = pos(e);
  stroke(last, p);
  last = p;
});

function endStroke() {
  if (!drawing) return;
  drawing = false;
  timer = setTimeout(predict, PREDICT_DELAY_MS);
}
pad.addEventListener("pointerup", endStroke);
pad.addEventListener("pointercancel", endStroke);

async function predict() {
  if (!hasInk) return;
  const id = ++reqId;
  statusEl.textContent = "predicting...";
  const blob = await new Promise((resolve) => pad.toBlob(resolve, "image/png"));
  const body = new FormData();
  body.append("file", blob, "drawing.png");
  try {
    const res = await fetch(`${window.API_BASE}/predict/${modelSel.value}`, { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || res.statusText);
    }
    const r = await res.json();
    if (id !== reqId) return; // cleared or redrawn while waiting
    textEl.textContent = r.text || "-";
    top3El.textContent = r.detections
      .map((d) => `${d.label} ${(d.confidence * 100).toFixed(0)}%`).join(", ") || "-";
    latencyEl.textContent = r.latency_ms.toFixed(1);
    statusEl.textContent = "done";
  } catch (e) {
    if (id === reqId) statusEl.textContent = `error: ${e.message}`;
  }
}

clearBtn.onclick = clearPad;
modelSel.onchange = () => { if (hasInk) predict(); };
clearPad();
