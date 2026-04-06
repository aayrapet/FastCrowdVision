let model = null;
let detectionInterval = null;
let isDetecting = false;

const videoUpload = document.getElementById("videoUpload");
const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const hiddenCanvas = document.getElementById("hiddenCanvas");
const hiddenCtx = hiddenCanvas.getContext("2d");

const playPauseBtn = document.getElementById("playPauseBtn");
const startDetectionBtn = document.getElementById("startDetectionBtn");
const stopDetectionBtn = document.getElementById("stopDetectionBtn");

const modelStatus = document.getElementById("modelStatus");
const detectionStatus = document.getElementById("detectionStatus");
const personCount = document.getElementById("personCount");

const scoreThresholdInput = document.getElementById("scoreThreshold");
const scoreThresholdValue = document.getElementById("scoreThresholdValue");

const intervalRange = document.getElementById("intervalRange");
const intervalValue = document.getElementById("intervalValue");

scoreThresholdInput.addEventListener("input", () => {
  scoreThresholdValue.textContent = Number(scoreThresholdInput.value).toFixed(2);
});

intervalRange.addEventListener("input", () => {
  intervalValue.textContent = intervalRange.value;
  if (isDetecting) {
    restartDetection();
  }
});

async function loadModel() {
  try {
    modelStatus.textContent = "Chargement du modèle...";
    model = await cocoSsd.load({
      base: "lite_mobilenet_v2"
    });
    modelStatus.textContent = "Modèle chargé";
  } catch (error) {
    console.error("Erreur lors du chargement du modèle :", error);
    modelStatus.textContent = "Erreur de chargement";
  }
}

function resizeCanvases() {
  const rect = video.getBoundingClientRect();

  overlay.width = rect.width;
  overlay.height = rect.height;
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;

  hiddenCanvas.width = video.videoWidth;
  hiddenCanvas.height = video.videoHeight;
}

function clearOverlay() {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawPredictions(predictions) {
  clearOverlay();

  const scaleX = overlay.width / video.videoWidth;
  const scaleY = overlay.height / video.videoHeight;

  overlayCtx.lineWidth = 2;
  overlayCtx.font = "16px Arial";

  predictions.forEach((pred) => {
    const [x, y, width, height] = pred.bbox;

    const drawX = x * scaleX;
    const drawY = y * scaleY;
    const drawW = width * scaleX;
    const drawH = height * scaleY;

    overlayCtx.strokeStyle = "#00FF7F";
    overlayCtx.fillStyle = "#00FF7F";

    overlayCtx.strokeRect(drawX, drawY, drawW, drawH);

    const text = `person ${(pred.score * 100).toFixed(1)}%`;
    const textWidth = overlayCtx.measureText(text).width;
    const textHeight = 20;

    overlayCtx.fillRect(drawX, Math.max(0, drawY - textHeight), textWidth + 10, textHeight);
    overlayCtx.fillStyle = "#000000";
    overlayCtx.fillText(text, drawX + 5, Math.max(15, drawY - 5));
  });
}

async function detectFrame() {
  if (!model || video.paused || video.ended || video.readyState < 2) {
    return;
  }

  try {
    hiddenCtx.drawImage(video, 0, 0, hiddenCanvas.width, hiddenCanvas.height);

    const predictions = await model.detect(hiddenCanvas);

    const threshold = Number(scoreThresholdInput.value);

    const personPredictions = predictions.filter(
      (pred) => pred.class === "person" && pred.score >= threshold
    );

    personCount.textContent = personPredictions.length.toString();
    drawPredictions(personPredictions);
  } catch (error) {
    console.error("Erreur pendant la détection :", error);
  }
}

function startDetection() {
  if (!model || isDetecting) return;

  isDetecting = true;
  detectionStatus.textContent = "Active";

  const intervalMs = Number(intervalRange.value);

  detectionInterval = setInterval(() => {
    detectFrame();
  }, intervalMs);

  startDetectionBtn.disabled = true;
  stopDetectionBtn.disabled = false;
}

function stopDetection() {
  isDetecting = false;
  detectionStatus.textContent = "Inactive";

  if (detectionInterval) {
    clearInterval(detectionInterval);
    detectionInterval = null;
  }

  clearOverlay();
  personCount.textContent = "0";

  startDetectionBtn.disabled = false;
  stopDetectionBtn.disabled = true;
}

function restartDetection() {
  stopDetection();
  startDetection();
}

videoUpload.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const videoURL = URL.createObjectURL(file);
  video.src = videoURL;

  playPauseBtn.disabled = false;
  startDetectionBtn.disabled = false;
});

video.addEventListener("loadedmetadata", () => {
  resizeCanvases();
});

video.addEventListener("play", () => {
  resizeCanvases();
});

window.addEventListener("resize", () => {
  if (video.videoWidth > 0) {
    resizeCanvases();
    clearOverlay();
  }
});

playPauseBtn.addEventListener("click", () => {
  if (video.paused) {
    video.play();
    playPauseBtn.textContent = "Pause";
  } else {
    video.pause();
    playPauseBtn.textContent = "Lecture";
  }
});

startDetectionBtn.addEventListener("click", () => {
  startDetection();
});

stopDetectionBtn.addEventListener("click", () => {
  stopDetection();
});

video.addEventListener("pause", () => {
  playPauseBtn.textContent = "Lecture";
});

video.addEventListener("ended", () => {
  playPauseBtn.textContent = "Lecture";
  stopDetection();
});

loadModel();