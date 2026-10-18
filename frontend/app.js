const SEND_INTERVAL_MS = 200; // ~5 frames per second
const MAX_WIDTH = 640;

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const modelSel = document.getElementById("model");
const toggleBtn = document.getElementById("toggle");
const statusEl = document.getElementById("status");
const textEl = document.getElementById("text");
const latencyEl = document.getElementById("latency");
const fpsEl = document.getElementById("fps");

const grab = document.createElement("canvas");
let ws = null, timer = null, stream = null, busy = false, lastReply = 0;

const wsBase = () => window.API_BASE.replace(/^http/, "ws");

function connect() {
  if (ws) ws.close();
  ws = new WebSocket(`${wsBase()}/ws/${modelSel.value}`);
  ws.onopen = () => (statusEl.textContent = "connected");
  ws.onclose = () => { statusEl.textContent = "disconnected"; busy = false; };
  ws.onerror = () => (statusEl.textContent = "connection error");
  ws.onmessage = (ev) => {
    busy = false;
    const r = JSON.parse(ev.data);
    if (r.error) { statusEl.textContent = r.error; return; }
    statusEl.textContent = "connected";
    textEl.textContent = r.text || "-";
    latencyEl.textContent = r.latency_ms.toFixed(1);
    const now = performance.now();
    if (lastReply) fpsEl.textContent = (1000 / (now - lastReply)).toFixed(1);
    lastReply = now;
    draw(r);
  };
}

function draw(r) {
  const ctx = overlay.getContext("2d");
  overlay.width = video.clientWidth;
  overlay.height = video.clientHeight;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const sx = overlay.width / r.image_size[0];
  const sy = overlay.height / r.image_size[1];
  ctx.lineWidth = 2;
  ctx.font = "16px system-ui";
  for (const d of r.detections) {
    if (!d.box) continue;
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeStyle = "#00e676";
    ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
    const label = `${d.label} ${(d.confidence * 100).toFixed(0)}%`;
    const w = ctx.measureText(label).width + 8;
    ctx.fillStyle = "#00e676";
    ctx.fillRect(x1 * sx, Math.max(0, y1 * sy - 22), w, 22);
    ctx.fillStyle = "#000";
    ctx.fillText(label, x1 * sx + 4, Math.max(16, y1 * sy - 6));
  }
}

function sendFrame() {
  if (busy || !ws || ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;
  const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
  grab.width = Math.round(video.videoWidth * scale);
  grab.height = Math.round(video.videoHeight * scale);
  grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);
  busy = true;
  ws.send(grab.toDataURL("image/jpeg", 0.7));
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
  } catch (e) {
    statusEl.textContent = "camera blocked (HTTPS or permission needed)";
    return;
  }
  video.srcObject = stream;
  connect();
  timer = setInterval(sendFrame, SEND_INTERVAL_MS);
  toggleBtn.textContent = "Stop camera";
}

function stop() {
  clearInterval(timer);
  if (ws) ws.close();
  if (stream) stream.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  overlay.getContext("2d").clearRect(0, 0, overlay.width, overlay.height);
  toggleBtn.textContent = "Start camera";
  statusEl.textContent = "idle";
  stream = null;
}

toggleBtn.onclick = () => (stream ? stop() : start());
modelSel.onchange = () => { if (stream) connect(); };
