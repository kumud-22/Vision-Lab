
import { connectors, webrtc, streams } from "@roboflow/inference-sdk";

/* =========================================================
   AI VISION LAB — APPLICATION
   API credentials remain on the backend.
   ========================================================= */

const $ = (id) => document.getElementById(id);

const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const statusEl = $("status");
const videoEl = $("video");
const feedEl = $("feed");
const placeholderEl = $("feedPlaceholder");
const dataPreviewEl = $("dataPreview");
const dataCountEl = $("dataCount");

const detectionListEl = $("detectionsList");
const visibleCountEl = $("visibleCount");
const historyCountEl = $("historyCount");
const historyListEl = $("historyList");
const thresholdEl = $("confidenceThreshold");
const thresholdValueEl = $("thresholdValue");
const currentThresholdEl = $("currentThreshold");
const themeSelectEl = $("themeSelect");
const layoutSelectEl = $("layoutSelect");
const settingsMessageEl = $("settingsMessage");

const configInputs = {
  workspaceName: $("workspaceName"),
  workflowId: $("workflowId"),
  imageInputName: $("imageInputName"),
  streamOutputNames: $("streamOutputNames"),
  dataOutputNames: $("dataOutputNames")
};

const exampleInputs = {
  streamOutput: $("exampleStreamOutput")
};

const serverInputs = {
  requestedRegion: $("requestedRegion"),
  requestedPlan: $("requestedPlan"),
  processingTimeout: $("processingTimeout")
};

const cameraInputs = {
  cameraSelect: $("cameraSelect"),
  resolutionSelect: $("resolutionSelect"),
  fpsSelect: $("fpsSelect"),
  cameraCaps: $("cameraCaps")
};

const tabBtns = document.querySelectorAll(".tab-btn");

let workflowMode = "example";
let activeConnection = null;
let cameraCapabilities = null;
let dataMessageCount = 0;
let confidenceThreshold = 0.70;
let detectionHistory = [];
let lastHistorySignature = "";
let lastHistoryTime = 0;
let latestPredictions = [];

/* Demo workflow: explicitly returns individual predictions. */
const WORKFLOW_SPEC = {
  version: "1.0",
  inputs: [
    { type: "InferenceImage", name: "image" }
  ],
  steps: [
    {
      type: "roboflow_core/roboflow_object_detection_model@v2",
      name: "model",
      images: "$inputs.image",
      model_id: "rfdetr-nano"
    },
    {
      type: "roboflow_core/bounding_box_visualization@v1",
      name: "boxes",
      image: "$inputs.image",
      predictions: "$steps.model.predictions"
    },
    {
      type: "roboflow_core/label_visualization@v1",
      name: "labels",
      image: "$steps.boxes.image",
      predictions: "$steps.model.predictions"
    },
    {
      type: "roboflow_core/property_definition@v1",
      name: "count",
      data: "$steps.model.predictions",
      operations: [
        { type: "SequenceLength" }
      ]
    }
  ],
  outputs: [
    {
      type: "JsonField",
      name: "labels",
      coordinates_system: "own",
      selector: "$steps.labels.image"
    },
    {
      type: "JsonField",
      name: "predictions",
      coordinates_system: "own",
      selector: "$steps.model.predictions"
    },
    {
      type: "JsonField",
      name: "count",
      coordinates_system: "own",
      selector: "$steps.count.output"
    }
  ]
};

/* -------------------- Configuration -------------------- */

function getServerConfig() {
  return {
    requestedRegion: serverInputs.requestedRegion?.value || "us",
    requestedPlan: serverInputs.requestedPlan?.value || "webrtc-gpu-small",
    processingTimeout:
      Number.parseInt(serverInputs.processingTimeout?.value, 10) || 600
  };
}

function getConfig() {
  const serverConfig = getServerConfig();

  if (workflowMode === "example") {
    const streamOutput = exampleInputs.streamOutput?.value || "output_image";

    return {
      mode: "example",
      workflowSpec: WORKFLOW_SPEC,
      imageInputName: "image",
      streamOutputNames: [streamOutput],
      dataOutputNames: ["predictions", "count"],
      ...serverConfig
    };
  }

  const workspaceName = configInputs.workspaceName?.value?.trim();
  const workflowId = configInputs.workflowId?.value?.trim();

  if (!workspaceName || !workflowId) {
    throw new Error("Enter your Roboflow workspace name and workflow ID in Dashboard.");
  }

  return {
    mode: "custom",
    workspaceName,
    workflowId,
    imageInputName: configInputs.imageInputName?.value?.trim() || "image",
    streamOutputNames: (
  configInputs.streamOutputNames?.value?.trim() || "output_image"
).split(",").map(value => value.trim()).filter(Boolean),
    dataOutputNames: (
      configInputs.dataOutputNames?.value?.trim() || "predictions"
    ).split(",").map(value => value.trim()).filter(Boolean),
    ...serverConfig
  };
}

function switchTab(tabName) {
  workflowMode = tabName;

  tabBtns.forEach(button => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });

  document.querySelectorAll(".tab-content").forEach(panel => {
    panel.hidden = panel.id !== `tab-${tabName}`;
  });
}

/* -------------------- Camera configuration -------------------- */

function getCameraConfig() {
  const resolution = cameraInputs.resolutionSelect?.value?.split("x") || [];

  return {
    deviceId: cameraInputs.cameraSelect?.value || undefined,
    width: Number.parseInt(resolution[0], 10) || 1280,
    height: Number.parseInt(resolution[1], 10) || 720,
    frameRate: Number.parseInt(cameraInputs.fpsSelect?.value, 10) || 30
  };
}

async function enumerateCameras() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraInputs.cameraSelect.innerHTML =
      '<option value="">Camera API unavailable</option>';
    cameraInputs.cameraCaps.textContent =
      "Open this site over HTTPS and allow camera access.";
    return;
  }

  try {
    const temporaryStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    temporaryStream.getTracks().forEach(track => track.stop());

    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(device => device.kind === "videoinput");

    cameraInputs.cameraSelect.replaceChildren();

    if (!cameras.length) {
      cameraInputs.cameraSelect.innerHTML =
        '<option value="">No cameras found</option>';
      return;
    }

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      cameraInputs.cameraSelect.appendChild(option);
    });

    cameraInputs.cameraSelect.disabled = false;

    await getCameraCapabilities(cameraInputs.cameraSelect.value);
  } catch (error) {
    console.error("[Camera] Enumeration failed:", error);
    cameraInputs.cameraSelect.innerHTML =
      '<option value="">Camera access denied</option>';
    cameraInputs.cameraCaps.textContent =
      "Allow camera access in your browser, then refresh.";
  }
}

async function getCameraCapabilities(deviceId) {
  if (!deviceId) return;

  cameraInputs.cameraCaps.textContent = "Checking camera capabilities…";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { deviceId: { exact: deviceId } },
      audio: false
    });

    const track = stream.getVideoTracks()[0];
    const capabilities = track.getCapabilities?.() || {};
    const settings = track.getSettings?.() || {};

    cameraCapabilities = capabilities;
    stream.getTracks().forEach(trackItem => trackItem.stop());

    populateResolutions(capabilities, settings);
    populateFrameRates(capabilities, settings);

    const details = [];
    if (capabilities.width && capabilities.height) {
      details.push(
        `${capabilities.width.min}–${capabilities.width.max} × ` +
        `${capabilities.height.min}–${capabilities.height.max}`
      );
    }
    if (capabilities.frameRate) {
      details.push(
        `${capabilities.frameRate.min}–${capabilities.frameRate.max} fps`
      );
    }

    cameraInputs.cameraCaps.textContent =
      details.join(" · ") || "Camera capabilities available.";
  } catch (error) {
    console.warn("[Camera] Capability check failed:", error);
    setDefaultCameraOptions();
    cameraInputs.cameraCaps.textContent =
      "Using standard camera settings.";
  }
}

function populateResolutions(capabilities, currentSettings) {
  const common = [
    [1920, 1080, "1080p"],
    [1280, 720, "720p"],
    [854, 480, "480p"],
    [640, 480, "VGA"],
    [640, 360, "360p"],
    [320, 240, "QVGA"]
  ];

  const maxW = capabilities.width?.max || 1920;
  const maxH = capabilities.height?.max || 1080;
  const minW = capabilities.width?.min || 320;
  const minH = capabilities.height?.min || 240;

  const available = common.filter(([w, h]) =>
    w >= minW && w <= maxW && h >= minH && h <= maxH
  );

  if (!available.length) available.push([640, 480, "VGA"]);

  cameraInputs.resolutionSelect.replaceChildren();

  available.forEach(([width, height, label]) => {
    const option = document.createElement("option");
    option.value = `${width}x${height}`;
    option.textContent = `${label} (${width}×${height})`;
    cameraInputs.resolutionSelect.appendChild(option);
  });

  const current = `${currentSettings.width || 0}x${currentSettings.height || 0}`;

  if ([...cameraInputs.resolutionSelect.options].some(option => option.value === current)) {
    cameraInputs.resolutionSelect.value = current;
  } else {
    cameraInputs.resolutionSelect.value =
      [...cameraInputs.resolutionSelect.options].find(option => option.value === "1280x720")
        ?.value || cameraInputs.resolutionSelect.options[0].value;
  }

  cameraInputs.resolutionSelect.disabled = false;
}

function populateFrameRates(capabilities, currentSettings) {
  const max = capabilities.frameRate?.max || 30;
  const min = capabilities.frameRate?.min || 1;

  const rates = [60, 30, 24, 15, 10].filter(rate =>
    rate >= min && rate <= max
  );

  if (!rates.length) rates.push(Math.floor(max) || 30);

  cameraInputs.fpsSelect.replaceChildren();

  rates.forEach(rate => {
    const option = document.createElement("option");
    option.value = String(rate);
    option.textContent = `${rate} fps`;
    cameraInputs.fpsSelect.appendChild(option);
  });

  const current = String(Math.round(currentSettings.frameRate || 30));
  cameraInputs.fpsSelect.value =
    rates.includes(Number(current)) ? current :
    rates.includes(30) ? "30" :
    String(rates[0]);

  cameraInputs.fpsSelect.disabled = false;
}

function setDefaultCameraOptions() {
  cameraInputs.resolutionSelect.innerHTML = `
    <option value="1280x720">720p (1280×720)</option>
    <option value="640x480">VGA (640×480)</option>
  `;

  cameraInputs.fpsSelect.innerHTML = `
    <option value="30">30 fps</option>
    <option value="24">24 fps</option>
    <option value="15">15 fps</option>
  `;

  cameraInputs.resolutionSelect.disabled = false;
  cameraInputs.fpsSelect.disabled = false;
}

/* -------------------- Detection parsing -------------------- */

function getConfidence(prediction) {
  const value = Number(
    prediction.confidence ??
    prediction.score ??
    prediction.probability
  );

  if (!Number.isFinite(value)) return null;
  return value > 1 ? value / 100 : value;
}

function getClassName(prediction) {
  return prediction.class ??
    prediction.class_name ??
    prediction.label ??
    prediction.name ??
    prediction.category_name ??
    null;
}

function extractPredictions(data) {
  const found = [];
  const visited = new Set();

  function walk(value, depth = 0) {
    if (!value || typeof value !== "object" || depth > 10) return;
    if (visited.has(value)) return;
    visited.add(value);

    if (Array.isArray(value)) {
      value.forEach(item => walk(item, depth + 1));
      return;
    }

    const label = getClassName(value);
    const confidence = getConfidence(value);

    if (label && confidence !== null) {
      found.push({
        label: String(label),
        confidence,
        raw: value
      });
    }

    for (const [key, child] of Object.entries(value)) {
      if (
        child && typeof child === "object" &&
        !["image", "output_image", "visualization"].includes(key)
      ) {
        walk(child, depth + 1);
      }
    }
  }

  walk(data);

  const unique = new Map();

  found.forEach(prediction => {
    const key = [
      prediction.label,
      prediction.confidence.toFixed(4),
      prediction.raw.x ?? "",
      prediction.raw.y ?? ""
    ].join(":");

    if (!unique.has(key)) unique.set(key, prediction);
  });

  return [...unique.values()];
}

function renderDetections(predictions) {
  latestPredictions = predictions;

  const visible = predictions
    .filter(item => item.confidence >= confidenceThreshold)
    .sort((a, b) => b.confidence - a.confidence);

  detectionListEl.replaceChildren();
  visibleCountEl.textContent = String(visible.length);

  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "detection-empty";
    empty.textContent = predictions.length
      ? "Detections are below your selected confidence threshold."
      : "No individual predictions received yet. Check your workflow output.";
    detectionListEl.appendChild(empty);
    return;
  }

  visible.forEach(prediction => {
    const card = document.createElement("article");
    card.className = "detection-card";

    const name = document.createElement("span");
    name.className = "detection-name";
    name.textContent = prediction.label;

    const confidence = document.createElement("span");
    confidence.className = "detection-confidence";
    confidence.textContent = `${Math.round(prediction.confidence * 100)}%`;

    const meter = document.createElement("div");
    meter.className = "detection-meter";

    const fill = document.createElement("div");
    fill.className = "detection-meter-fill";
    fill.style.width = `${Math.round(prediction.confidence * 100)}%`;

    meter.appendChild(fill);
    card.append(name, confidence, meter);
    detectionListEl.appendChild(card);
  });
}

function recordHistory(predictions) {
  if (!predictions.length) return;

  const signature = predictions
    .map(item => `${item.label}:${item.confidence.toFixed(2)}`)
    .sort()
    .join("|");

  const now = Date.now();

  if (
    signature === lastHistorySignature &&
    now - lastHistoryTime < 1000
  ) {
    return;
  }

  lastHistorySignature = signature;
  lastHistoryTime = now;

  detectionHistory.unshift({
    time: new Date().toLocaleTimeString(),
    items: predictions.map(item => ({
      label: item.label,
      confidence: item.confidence
    }))
  });

  detectionHistory = detectionHistory.slice(0, 100);
  renderHistory();
}

function renderHistory() {
  historyListEl.replaceChildren();

  historyCountEl.textContent = String(detectionHistory.length);

  if (!detectionHistory.length) {
    const empty = document.createElement("p");
    empty.className = "detection-empty";
    empty.textContent = "No detection events recorded yet.";
    historyListEl.appendChild(empty);
    return;
  }

  detectionHistory.forEach(entry => {
    const row = document.createElement("article");
    row.className = "history-entry";

    const time = document.createElement("span");
    time.textContent = entry.time;

    const items = document.createElement("span");
    items.textContent = entry.items.map(item =>
      `${item.label} (${Math.round(item.confidence * 100)}%)`
    ).join(", ");

    row.append(time, items);
    historyListEl.appendChild(row);
  });
}

/* -------------------- Confidence and appearance -------------------- */

function updateThreshold(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return;

  confidenceThreshold = Math.max(0, Math.min(1, parsed / 100));

  const display = `${Math.round(confidenceThreshold * 100)}%`;
  thresholdValueEl.textContent = display;
  currentThresholdEl.textContent = display;

  renderDetections(latestPredictions);
}

function applyAppearance() {
  const theme = themeSelectEl.value || "cyan";
  const layout = layoutSelectEl.value || "standard";

  document.documentElement.dataset.theme = theme;
  document.documentElement.dataset.layout = layout;

  try {
    localStorage.setItem("visionLabTheme", theme);
    localStorage.setItem("visionLabLayout", layout);
  } catch (error) {
    console.warn("[Settings] Browser storage unavailable:", error);
  }
}

function loadAppearance() {
  try {
    themeSelectEl.value = localStorage.getItem("visionLabTheme") || "cyan";
    layoutSelectEl.value = localStorage.getItem("visionLabLayout") || "standard";
  } catch {
    themeSelectEl.value = "cyan";
    layoutSelectEl.value = "standard";
  }

  applyAppearance();
}

function clearHistory() {
  detectionHistory = [];
  lastHistorySignature = "";
  lastHistoryTime = 0;
  renderHistory();
}

$("saveSettingsBtn").addEventListener("click", () => {
  applyAppearance();
  settingsMessageEl.textContent = "Display preferences saved in this browser.";
  window.setTimeout(() => {
    settingsMessageEl.textContent = "";
  }, 3000);
});

$("resetHistoryBtn").addEventListener("click", clearHistory);
$("clearHistoryBtn").addEventListener("click", clearHistory);

thresholdEl.addEventListener("input", () => {
  updateThreshold(thresholdEl.value);
});

themeSelectEl.addEventListener("change", applyAppearance);
layoutSelectEl.addEventListener("change", applyAppearance);

/* -------------------- Secure WebRTC connection -------------------- */

function setStatus(message) {
  statusEl.textContent = message;
  console.log("[Status]", message);
}

async function connectWebcamToRoboflowWebRTC({ onData } = {}) {
  const config = getConfig();
  const cameraConfig = getCameraConfig();

  const connector = connectors.withProxyUrl("/api/init-webrtc", {
    turnConfigUrl: "/api/turn-config"
  });

  const videoConstraints = {
    width: { ideal: cameraConfig.width },
    height: { ideal: cameraConfig.height },
    frameRate: {
      ideal: cameraConfig.frameRate,
      max: cameraConfig.frameRate
    }
  };

  if (cameraConfig.deviceId) {
    videoConstraints.deviceId = { exact: cameraConfig.deviceId };
  } else {
    videoConstraints.facingMode = { ideal: "environment" };
  }

  const baseParams = {
    imageInputName: config.imageInputName,
    streamOutputNames: config.streamOutputNames,
    dataOutputNames: config.dataOutputNames,
    requestedRegion: config.requestedRegion,
    requestedPlan: config.requestedPlan,
    processingTimeout: config.processingTimeout
  };

  const wrtcParams = config.mode === "example"
    ? { ...baseParams, workflowSpec: config.workflowSpec }
    : {
        ...baseParams,
        workspaceName: config.workspaceName,
        workflowId: config.workflowId
      };

  const source = await streams.useCamera({
    video: videoConstraints,
    audio: false
  });

  return webrtc.useStream({
    source,
    connector,
    wrtcParams,
    onData,
    options: {
      disableInputStreamDownscaling: true
    }
  });
}

async function start() {
  if (activeConnection) return;

  startBtn.disabled = true;
  setStatus("Connecting…");

  try {
    const connection = await connectWebcamToRoboflowWebRTC({
      onData(data) {
        console.log("[Roboflow data]", data);

        dataMessageCount += 1;
        dataCountEl.textContent = String(dataMessageCount);
        dataPreviewEl.textContent = JSON.stringify(data, null, 2);

        const predictions = extractPredictions(data);
        renderDetections(predictions);

        recordHistory(
          predictions.filter(item => item.confidence >= confidenceThreshold)
        );
      }
    });

    activeConnection = connection;

    const remoteStream = await connection.remoteStream();
    videoEl.srcObject = remoteStream;
    videoEl.muted = true;
    videoEl.playsInline = true;

    try {
      await videoEl.play();
    } catch (error) {
      console.warn("[Video] Autoplay was blocked:", error);
    }

    setStatus("Connected · Processing video");
    stopBtn.disabled = false;
    placeholderEl.classList.add("hidden");
    feedEl.classList.add("scanning");
  } catch (error) {
    console.error("[Connection] Failed:", error);

    const message = error?.message || String(error);
    setStatus(`Error: ${message}`);

    if (/api key/i.test(message)) {
      alert("The server API key is not configured. Check your backend environment settings.");
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    activeConnection = null;
  }
}

async function stop() {
  if (!activeConnection) return;

  stopBtn.disabled = true;
  setStatus("Stopping…");

  try {
    await activeConnection.cleanup();
  } catch (error) {
    console.error("[Connection] Cleanup failed:", error);
  } finally {
    activeConnection = null;
    videoEl.srcObject = null;
    placeholderEl.classList.remove("hidden");
    feedEl.classList.remove("scanning");

    startBtn.disabled = false;
    stopBtn.disabled = true;

    dataMessageCount = 0;
    dataCountEl.textContent = "0";
    dataPreviewEl.textContent = "";

    latestPredictions = [];
    renderDetections([]);

    setStatus("Idle");
  }
}

/* -------------------- Event listeners -------------------- */

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);

tabBtns.forEach(button => {
  button.addEventListener("click", () => {
    switchTab(button.dataset.tab);
  });
});

cameraInputs.cameraSelect.addEventListener("change", async () => {
  await getCameraCapabilities(cameraInputs.cameraSelect.value);
});

window.addEventListener("pagehide", () => {
  if (activeConnection) {
    activeConnection.cleanup();
  }
});

/* -------------------- Initialization -------------------- */

loadAppearance();
updateThreshold(thresholdEl.value);
renderHistory();
setDefaultCameraOptions();

enumerateCameras();

fetch("/api/health")
  .then(response => {
    if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
    return response.json();
  })
  .then(data => {
    console.log("[Server health]", data);
    if (!data.apiKeyConfigured) {
      console.warn("[Server] API key is not configured.");
    }
  })
  .catch(error => {
    console.warn("[Server] Health check unavailable:", error);
  });
