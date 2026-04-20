// app.js — Frontend for FastCrowdVision.
// WebSocket communication to the Python backend (server-side SSD + norfair tracking).
// sends results, and the browser seeks the video to match.

// ── DOM elements ────────────────────────────────────────────────────

const videoUpload = document.getElementById("videoUpload");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const startDetectionBtn = document.getElementById("startDetectionBtn");
const stopDetectionBtn = document.getElementById("stopDetectionBtn");

const serverStatus = document.getElementById("serverStatus");
const detectionStatus = document.getElementById("detectionStatus");
const currentCount = document.getElementById("currentCount");
const totalUnique = document.getElementById("totalUnique");
const progressInfo = document.getElementById("progressInfo");

const scoreThresholdInput = document.getElementById("scoreThreshold");
const scoreThresholdValue = document.getElementById("scoreThresholdValue");
const frameSkipRange = document.getElementById("frameSkipRange");
const frameSkipValue = document.getElementById("frameSkipValue");

// ── State ───────────────────────────────────────────────────────────

let sessionId = null;
let ws = null;
let selectedFile = null;
let videoMeta = null;
let currentVideoURL = null;

// ── Slider listeners ────────────────────────────────────────────────

scoreThresholdInput.addEventListener("input", () => {
  scoreThresholdValue.textContent = Number(scoreThresholdInput.value).toFixed(2);
});

frameSkipRange.addEventListener("input", () => {
  frameSkipValue.textContent = frameSkipRange.value;
});

// ── Canvas helpers ──────────────────────────────────────────────────

function resizeOverlay() {
  const rect = video.getBoundingClientRect();
  overlay.width = rect.width;
  overlay.height = rect.height;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

// ── Drawing ─────────────────────────────────────────────────────────

// CHANGED: now receives server JSON data with track_ids instead of
// TF.js prediction objects. Draws boxes with track ID labels.

// assign a stable color to each track ID so the same person keeps the same color
const trackColors = {};
const colorPalette = [
  "#00FF7F", "#FF6347", "#1E90FF", "#FFD700", "#FF69B4",
  "#00CED1", "#FF8C00", "#8A2BE2", "#32CD32", "#DC143C",
];

function getTrackColor(trackId) {
  if (!(trackId in trackColors)) {
    trackColors[trackId] = colorPalette[Object.keys(trackColors).length % colorPalette.length];
  }
  return trackColors[trackId];
}

function drawDetections(data) {
  clearOverlay();

  // scale factor: server sends pixel coords for the original video resolution,
  // but the canvas may be displayed at a different size
  const scaleX = overlay.width / video.videoWidth;
  const scaleY = overlay.height / video.videoHeight;

  overlayCtx.lineWidth = 2;
  overlayCtx.font = "14px Arial";

  for (let i = 0; i < data.boxes.length; i++) {
    const [x1, y1, x2, y2] = data.boxes[i];
    const trackId = data.track_ids[i];
    const score = data.scores[i];
    const cls = data.classes[i];
    const color = getTrackColor(trackId);

    const drawX = x1 * scaleX;
    const drawY = y1 * scaleY;
    const drawW = (x2 - x1) * scaleX;
    const drawH = (y2 - y1) * scaleY;

    // draw bounding box
    overlayCtx.strokeStyle = color;
    overlayCtx.strokeRect(drawX, drawY, drawW, drawH);

    // draw label background + text with track ID, class name, and score
    const text = `#${trackId} ${cls} ${(score * 100).toFixed(0)}%`;
    const textWidth = overlayCtx.measureText(text).width;
    const textHeight = 18;

    overlayCtx.fillStyle = color;
    overlayCtx.fillRect(drawX, Math.max(0, drawY - textHeight), textWidth + 8, textHeight);
    overlayCtx.fillStyle = "#000000";
    overlayCtx.fillText(text, drawX + 4, Math.max(14, drawY - 4));
  }
}

// ── Upload ──────────────────────────────────────────────────────────

// ADDED: upload the video file to the server via POST /upload

async function uploadVideo(file) {
  serverStatus.textContent = "Envoi de la vidéo...";

  const formData = new FormData();
  formData.append("file", file);

  const resp = await fetch("/upload", { method: "POST", body: formData });
  if (!resp.ok) {
    serverStatus.textContent = "Erreur d'envoi";
    throw new Error("Upload failed");
  }

  const result = await resp.json();
  serverStatus.textContent = "Vidéo reçue par le serveur";
  return result.session_id;
}

// ── WebSocket detection ─────────────────────────────────────────────

// ADDED: open a WebSocket to /ws/detect, send config, receive frame-by-frame results

function startDetection() {
  if (!sessionId) return;

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/detect`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    detectionStatus.textContent = "Active — traitement en cours...";
    startDetectionBtn.disabled = true;
    stopDetectionBtn.disabled = false;
    ws.send(JSON.stringify({
      session_id: sessionId,
      score_thr: Number(scoreThresholdInput.value),
      frame_skip: Number(frameSkipRange.value),
    }));
  };

  let lastFrameTime = null;

  ws.onmessage = (event) => {
    const now = performance.now();
    const data = JSON.parse(event.data);   // ← un seul parse

    if (data.type === "metadata") {
      videoMeta = data;
      return;
    }

    if (data.type === "detection") {
      // mesure du throughput
      if (lastFrameTime) {
        const fps = 1000 / (now - lastFrameTime);
        console.log(`Throughput: ${fps.toFixed(1)} frames/s`);
      }
      lastFrameTime = now;

      if (videoMeta) video.currentTime = data.time;
      drawDetections(data);
      currentCount.textContent = data.current_count;
      totalUnique.textContent = data.total_unique;

      if (videoMeta && videoMeta.total_frames > 0) {
        const pct = Math.round((data.frame / videoMeta.total_frames) * 100);
        progressInfo.textContent = `Image ${data.frame} / ${videoMeta.total_frames} (${pct}%)`;
      }
      return;
    }

    if (data.type === "done") {
      detectionStatus.textContent = "Terminé";
      totalUnique.textContent = data.total_unique;
      progressInfo.textContent = "Terminé";
      stopDetectionBtn.disabled = true;
      startDetectionBtn.disabled = true;
      ws = null;
      sessionId = null;
      serverStatus.textContent = "Prêt — choisissez une nouvelle vidéo";
      return;
    }

    if (data.type === "error") {
      detectionStatus.textContent = `Erreur : ${data.message}`;
      return;
    }
  };

  ws.onclose = () => {
    if (detectionStatus.textContent !== "Terminé") {
      detectionStatus.textContent = "Connexion fermée";
      startDetectionBtn.disabled = !sessionId;
    }
    stopDetectionBtn.disabled = true;
  };

  ws.onerror = () => {
    detectionStatus.textContent = "Erreur WebSocket";
  };
}

function stopDetection() {
  if (ws) {
    ws.close();
    ws = null;
  }
  detectionStatus.textContent = "Arrêtée";
  clearOverlay();
  currentCount.textContent = "0";
  startDetectionBtn.disabled = false;
  stopDetectionBtn.disabled = true;
}

// ── Event listeners ─────────────────────────────────────────────────

// CHANGED: on file select, show local preview AND upload to server.
// Also resets all state from the previous video so the user doesn't
// need to reload the page.
videoUpload.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  selectedFile = file;

  // reset all state from the previous video
  if (ws) { ws.close(); ws = null; }
  sessionId = null;
  videoMeta = null;
  for (const key in trackColors) delete trackColors[key];
  clearOverlay();
  currentCount.textContent = "0";
  totalUnique.textContent = "0";
  progressInfo.textContent = "—";
  detectionStatus.textContent = "Envoi en cours…";
  startDetectionBtn.disabled = true;
  stopDetectionBtn.disabled = true;

  // revoke old blob URL to free memory, then create a new one
  if (currentVideoURL) URL.revokeObjectURL(currentVideoURL);
  currentVideoURL = URL.createObjectURL(file);
  video.src = currentVideoURL;
  video.load();

  // reset file input so the same file can be re-selected later
  event.target.value = "";

  // upload to server in the background
  try {
    sessionId = await uploadVideo(file);
    if (sessionId) {
      startDetectionBtn.disabled = false;
      detectionStatus.textContent = "Prêt — lancez la détection";
    }
  } catch (err) {
    console.error("Upload error:", err);
    detectionStatus.textContent = "Erreur lors de l'envoi";
  }
});

video.addEventListener("loadedmetadata", () => {
  resizeOverlay();
});

window.addEventListener("resize", () => {
  if (video.videoWidth > 0) {
    resizeOverlay();
    clearOverlay();
  }
});

startDetectionBtn.addEventListener("click", () => {
  startDetection();
});

stopDetectionBtn.addEventListener("click", () => {
  stopDetection();
});
