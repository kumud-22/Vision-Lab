import { connectors, webrtc, streams } from "@roboflow/inference-sdk";

// ------------------------------------------------------
// AI VISION LAB
// Uses your existing secure backend proxy.
// Never put a Roboflow API key in this frontend file.
// ------------------------------------------------------

const $ = (id) => document.getElementById(id);

let connection = null;
let localStream = null;
let remoteStream = null;
let running = false;
let messageCount = 0;
let latestPredictions = [];
let history = [];

const MAX_HISTORY = 30;

const themeClasses = [
  "theme-cyan",
  "theme-purple",
  "theme-green",
  "theme-pink",
  "theme-blue",
];

function setStatus(message, live = false) {
  $("status").textContent = message;
  $("statusPill").classList.toggle("live", live);
  $("feedStatus").textContent = message;
}

function setMessage(message) {
  $("settingsMessage").textContent = message;
}

function safeNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function formatConfidence(value) {
  let confidence = safeNumber(value);

  if (confidence === null) return null;

  // Most detection models return confidence between 0 and 1.
  if (confidence > 1 && confidence <= 100) {
    confidence /= 100;
  }

  return Math.max(0, Math.min(1, confidence));
}

function confidenceText(value) {
  return `${Math.round(value * 100)}%`;
}

// ------------------------------------------------------
// Parse prediction objects from the workflow data.
// Supports common prediction arrays and nested objects.
// ------------------------------------------------------

function findPredictionArray(value, depth = 0) {
  if (depth > 7 || value == null) return null;

  if (typeof value === "string") {
    try {
      return findPredictionArray(JSON.parse(value), depth + 1);
    } catch {
      return null;
    }
  }

  if (Array.isArray(value)) {
    const looksLikePredictions = value.some((item) => {
      if (!item || typeof item !== "object") return false;

      return (
        "class" in item ||
        "class_name" in item ||
        "label" in item ||
        "confidence" in item ||
        "class_id" in item
      );
    });

    if (looksLikePredictions) return value;

    for (const item of value) {
      const result = findPredictionArray(item, depth + 1);
      if (result) return result;
    }

    return null;
  }

  if (typeof value !== "object") return null;

  // Prioritise the expected data output.
  for (const key of [
    "predictions",
    "detections",
    "objects",
    "results",
  ]) {
    if (key in value) {
      const result = findPredictionArray(value[key], depth + 1);
      if (result) return result;
    }
  }

  for (const item of Object.values(value)) {
    if (item && typeof item === "object") {
      const result = findPredictionArray(item, depth + 1);
      if (result) return result;
    }
  }

  return null;
}

function normalisePredictions(data) {
  const rawArray = findPredictionArray(data);

  if (!rawArray) return [];

  return rawArray.map((item, index) => {
    const rawConfidence =
      item.confidence ??
      item.score ??
      item.probability ??
      item.prob;

    const confidence = formatConfidence(rawConfidence);

    const name =
      item.class ??
      item.class_name ??
      item.label ??
      item.name ??
      (item.class_id != null ? `Class ${item.class_id}` : "Unknown object");

    return {
      id: `${index}-${String(name)}`,
      name: String(name),
      confidence,
      raw: item,
    };
  }).filter((item) => item.confidence !== null);
}

// ------------------------------------------------------
// Display detections and client-side confidence filtering.
// This filter does not modify Roboflow's model settings.
// ------------------------------------------------------

function getThreshold() {
  return Number($("confidenceThreshold").value) / 100;
}

function renderDetections() {
  const threshold = getThreshold();

  $("thresholdValue").textContent =
    `${Math.round(threshold * 100)}%`;

  $("currentThreshold").textContent =
    `${Math.round(threshold * 100)}%`;

  const visible = latestPredictions.filter(
    (item) => item.confidence >= threshold
  );

  $("visibleCount").textContent = String(visible.length);

  const container = $("detectionsList");
  container.replaceChildren();

  if (!visible.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = latestPredictions.length
      ? "No detections meet the current display threshold."
      : "No prediction objects found in the latest data message. Open Raw Workflow Data to inspect the response.";
    container.appendChild(empty);
    return;
  }

  visible.forEach((item) => {
    const card = document.createElement("div");
    card.className = "detection-item";

    const top = document.createElement("div");
    top.className = "detection-top";

    const name = document.createElement("strong");
    name.textContent = item.name;

    const score = document.createElement("span");
    score.className = "confidence";
    score.textContent = confidenceText(item.confidence);

    top.append(name, score);

    const bar = document.createElement("div");
    bar.className = "bar";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${item.confidence * 100}%`;

    bar.appendChild(fill);
    card.append(top, bar);
    container.appendChild(card);
  });
}

function addHistory(data, predictions) {
  const visible = predictions.filter(
    (item) => item.confidence >= getThreshold()
  );

  history.unshift({
    time: new Date().toLocaleTimeString(),
    count: visible.length,
    names: visible.map((item) => item.name),
    data,
  });

  history = history.slice(0, MAX_HISTORY);

  $("historyCount").textContent = String(history.length);
  renderHistory();
}

function renderHistory() {
  const container = $("historyList");
  container.replaceChildren();

  if (!history.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "Your session history will appear here.";
    container.appendChild(empty);
    return;
  }

  history.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "history-item";

    const title = document.createElement("div");
    title.className = "detection-top";

    const time = document.createElement("strong");
    time.textContent = entry.time;

    const count = document.createElement("span");
    count.className = "confidence";
    count.textContent = `${entry.count} object(s)`;

    title.append(time, count);

    const description = document.createElement("div");
    description.className = "hint";
    description.textContent = entry.names.length
      ? entry.names.join(", ")
      : "No objects passed the display threshold.";

    card.append(title, description);
    container.appendChild(card);
  });
}

function handleData(data) {
  messageCount += 1;
  $("dataCount").textContent = String(messageCount);

  // Show the unmodified response to help diagnose workflow outputs.
  $("dataPreview").textContent = JSON.stringify(data, null, 2);

  latestPredictions = normalisePredictions(data);
  renderDetections();
  addHistory(data, latestPredictions);
}

// ------------------------------------------------------
// Camera setup
// ------------------------------------------------------

async function refreshCameras() {
  const select = $("cameraSelect");
  const previous = select.value;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter((device) => device.kind === "videoinput");

    select.replaceChildren();

    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Default camera";
    select.appendChild(defaultOption);

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      select.appendChild(option);
    });

    if ([...select.options].some((option) => option.value === previous)) {
      select.value = previous;
    }
  } catch (error) {
    console.warn("Could not list cameras:", error);
  }
}

function getVideoConstraints() {
  const [width, height] = $("resolutionSelect").value
    .split("x")
    .map(Number);

  const fps = Number($("fpsSelect").value);
  const deviceId = $("cameraSelect").value;

  const video = {
    width: { ideal: width },
    height: { ideal: height },
    frameRate: { ideal: fps, max: fps },
  };

  if (deviceId) {
    video.deviceId = { exact: deviceId };
  } else {
    video.facingMode = "environment";
  }

  return video;
}

// ------------------------------------------------------
// Workflow configuration
// IMPORTANT:
// streamOutputNames = output_image
// dataOutputNames   = predictions
// ------------------------------------------------------

function getWorkflowParams() {
  const workspaceName = $("workspaceName").value.trim();
  const workflowId = $("workflowId").value.trim();
  const imageInputName = $("imageInputName").value.trim() || "image";

  if (!workspaceName || !workflowId) {
    throw new Error("Enter your Roboflow workspace name and workflow ID.");
  }

  const params = {
    workspaceName,
    workflowId,
    imageInputName,

    // These are valid outputs reported by your workflow.
    // Do not change the stream output to "labels".
    streamOutputNames: ["output_image"],
    dataOutputNames: ["predictions"],
  };

  const region = $("requestedRegion").value.trim();
  const plan = $("requestedPlan").value.trim();
  const timeout = Number($("processingTimeout").value);

  if (region) params.requestedRegion = region;
  if (plan) params.requestedPlan = plan;
  if (Number.isFinite(timeout) && timeout > 0) {
    params.processingTimeout = timeout;
  }

  return params;
}

// ------------------------------------------------------
// Start and stop the real-time inference session
// ------------------------------------------------------

async function start() {
  if (running) return;

  $("startBtn").disabled = true;
  setStatus("Requesting camera access…");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access is unavailable. Open the site using HTTPS or localhost."
      );
    }

    const wrtcParams = getWorkflowParams();

    // Use the existing backend proxy, not an API key in the browser.
    const connector = connectors.withProxyUrl("/api/init-webrtc", {
      turnConfigUrl: "/api/turn-config",
    });

    localStream = await streams.useCamera({
      video: getVideoConstraints(),
      audio: false,
    });

    await refreshCameras();

    setStatus("Connecting to Roboflow…");

    connection = await webrtc.useStream({
      source: localStream,
      connector,
      wrtcParams,
      onData: handleData,
      options: {
        disableInputStreamDownscaling: true,
      },
    });

    remoteStream = await connection.remoteStream();

    $("video").srcObject = remoteStream;
    $("video").hidden = false;
    $("feedPlaceholder").hidden = true;

    await $("video").play().catch(() => {});

    running = true;
    $("stopBtn").disabled = false;
    $("startBtn").disabled = true;

    setStatus("Live inference active", true);
    $("feedStatus").textContent = "Receiving processed video";
  } catch (error) {
    console.error("Could not start inference:", error);

    await stop(false);

    setStatus("Connection failed");
    $("feedStatus").textContent = "Check the error message below.";

    const message = error?.message || String(error);
    alert(`Could not start the AI Vision Lab:\n\n${message}`);
  } finally {
    if (!running) $("startBtn").disabled = false;
  }
}

async function stop(updateStatus = true) {
  running = false;

  try {
    if (connection?.close) await connection.close();
    else if (connection?.stop) await connection.stop();
  } catch (error) {
    console.warn("Connection cleanup:", error);
  }

  connection = null;

  // Stop all local and remote media tracks.
  for (const stream of [localStream, remoteStream]) {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
  }

  localStream = null;
  remoteStream = null;

  $("video").pause();
  $("video").srcObject = null;
  $("video").hidden = true;
  $("feedPlaceholder").hidden = false;

  $("startBtn").disabled = false;
  $("stopBtn").disabled = true;

  if (updateStatus) {
    setStatus("Camera stopped");
    $("feedStatus").textContent = "Camera inactive";
  }
}

// ------------------------------------------------------
// Settings and event listeners
// ------------------------------------------------------

function applyTheme(theme) {
  document.body.classList.remove(...themeClasses);
  document.body.classList.add(`theme-${theme}`);
}

function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem("visionLabSettings") || "{}");

    if (saved.theme) $("themeSelect").value = saved.theme;
    if (saved.layout) $("layoutSelect").value = saved.layout;
    if (saved.threshold != null) {
      $("confidenceThreshold").value = saved.threshold;
    }
  } catch {
    // Ignore invalid or unavailable browser storage.
  }

  applyTheme($("themeSelect").value);
  document.body.classList.toggle(
    "compact",
    $("layoutSelect").value === "compact"
  );

  renderDetections();
}

$("startBtn").addEventListener("click", start);
$("stopBtn").addEventListener("click", () => stop());

$("confidenceThreshold").addEventListener("input", renderDetections);

$("themeSelect").addEventListener("change", (event) => {
  applyTheme(event.target.value);
});

$("layoutSelect").addEventListener("change", (event) => {
  document.body.classList.toggle(
    "compact",
    event.target.value === "compact"
  );
});

$("saveSettingsBtn").addEventListener("click", () => {
  const settings = {
    theme: $("themeSelect").value,
    layout: $("layoutSelect").value,
    threshold: Number($("confidenceThreshold").value),
  };

  try {
    localStorage.setItem("visionLabSettings", JSON.stringify(settings));
    setMessage("Display settings saved in this browser.");
  } catch {
    setMessage("Settings applied, but browser storage is unavailable.");
  }

  applyTheme(settings.theme);
  document.body.classList.toggle(
    "compact",
    settings.layout === "compact"
  );

  renderDetections();
});

$("refreshCamerasBtn").addEventListener("click", refreshCameras);

$("clearHistoryBtn").addEventListener("click", () => {
  history = [];
  $("historyCount").textContent = "0";
  renderHistory();
});

$("resetHistoryBtn").addEventListener("click", () => {
  history = [];
  messageCount = 0;
  latestPredictions = [];

  $("historyCount").textContent = "0";
  $("dataCount").textContent = "0";
  $("visibleCount").textContent = "0";
  $("dataPreview").textContent = "No data received yet.";

  renderHistory();
  renderDetections();
});

window.addEventListener("beforeunload", () => {
  for (const stream of [localStream, remoteStream]) {
    stream?.getTracks().forEach((track) => track.stop());
  }
});

loadSettings();
refreshCameras();
setStatus("Ready to initialise");
