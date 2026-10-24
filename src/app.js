
import { connectors, webrtc, streams } from "@roboflow/inference-sdk";

// ============================================================
// VISION LAB — APPLICATION CONTROLLER
// ============================================================

const $ = (id) => document.getElementById(id);

const els = {
  start: $("startBtn"),
  stop: $("stopBtn"),
  status: $("status"),
  video: $("video"),
  placeholder: $("feedPlaceholder"),
  feedState: $("feedState"),
  feedHint: $("feedHint"),
  feedDot: $("feedDot"),
  headerDot: $("headerDot"),
  systemDot: $("systemDot"),
  systemLabel: $("systemLabel"),
  detectionList: $("detectionList"),
  visibleCount: $("visibleCount"),
  totalDetections: $("totalDetections"),
  uniqueClasses: $("uniqueClasses"),
  thresholdMetric: $("thresholdMetric"),
  threshold: $("confidenceThreshold"),
  thresholdLabel: $("thresholdLabel"),
  thresholdHelp: $("thresholdHelp"),
  inferenceMode: $("inferenceMode"),
  modelId: $("modelId"),
  workspace: $("workspaceName"),
  workflowId: $("workflowId"),
  imageInput: $("imageInputName"),
  streamOutputs: $("streamOutputNames"),
  dataOutputs: $("dataOutputNames"),
  region: $("requestedRegion"),
  plan: $("requestedPlan"),
  timeout: $("processingTimeout"),
  modelIdField: $("modelIdField"),
  historyList: $("historyList"),
  historySearch: $("historySearch"),
  dashboardRecent: $("dashboardRecent"),
  dashEvents: $("dashEvents"),
  dashClasses: $("dashClasses"),
  dashThreshold: $("dashThreshold"),
  systemStream: $("systemStream"),
  backendState: $("backendState"),
  apiKeyState: $("apiKeyState"),
  density: $("densitySelect"),
  boxStyle: $("boxStyleSelect"),
  showSidebar: $("showSidebar")
};

const STORAGE = {
  settings: "vision-lab-settings-v1",
  pipeline: "vision-lab-pipeline-v1",
  history: "vision-lab-history-v1"
};

let connection = null;
let starting = false;
let stopping = false;
let reconnecting = false;
let eventCount = 0;
let lastDetections = [];
let sessionClasses = new Set();
let history = loadJSON(STORAGE.history, []);
let lastHealth = null;

// ============================================================
// SAFE LOCAL STORAGE
// ============================================================

function loadJSON(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
}

function saveJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) {
    console.warn("[Storage] Could not save:", error);
  }
}

// ============================================================
// STATUS
// ============================================================

function setStatus(message, state = "idle") {
  els.status.textContent = message;
  els.systemLabel.textContent =
    state === "live" ? "Inference active" :
    state === "connecting" ? "Connecting" :
    state === "error" ? "Check system" : "System ready";

  const colors = {
    live: "var(--green)",
    connecting: "var(--accent)",
    error: "var(--danger)",
    idle: "#71809a"
  };

  const color = colors[state] || colors.idle;

  [els.headerDot, els.systemDot, els.feedDot].forEach(dot => {
    if (!dot) return;
    dot.style.background = color;
    dot.style.boxShadow =
      state === "live" ? "0 0 12px var(--green)" : "none";
  });

  els.feedState.textContent =
    state === "live" ? "LIVE" :
    state === "connecting" ? "CONNECTING" :
    state === "error" ? "ERROR" : "STANDBY";

  if (els.systemStream) {
    els.systemStream.textContent =
      state === "live" ? "Live" :
      state === "connecting" ? "Connecting" :
      state === "error" ? "Error" : "Idle";
  }
}

function setButtonState() {
  els.start.disabled = starting || Boolean(connection);
  els.stop.disabled = !connection || stopping;
}

// ============================================================
// NAVIGATION
// ============================================================

const viewTitles = {
  vision: "Live Vision",
  dashboard: "Dashboard",
  history: "Reading History",
  pipeline: "Inference Pipeline",
  settings: "Settings",
  system: "System Status"
};

function showView(name) {
  if (!viewTitles[name]) return;

  document.querySelectorAll(".view").forEach(view => {
    view.classList.toggle("active", view.id === `view-${name}`);
  });

  document.querySelectorAll(".nav-btn").forEach(button => {
    button.classList.toggle("active", button.dataset.view === name);
  });

  $("pageTitle").textContent = viewTitles[name];
}

document.querySelectorAll(".nav-btn").forEach(button => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

document.querySelectorAll("[data-go]").forEach(button => {
  button.addEventListener("click", () => showView(button.dataset.go));
});

$("openPipelineBtn").addEventListener("click", () => showView("pipeline"));
$("openSettingsBtn").addEventListener("click", () => showView("settings"));

// ============================================================
// PIPELINE CONFIGURATION
// ============================================================

function readPipeline() {
  return {
    mode: els.inferenceMode.value,
    modelId: els.modelId.value.trim(),
    workspaceName: els.workspace.value.trim(),
    workflowId: els.workflowId.value.trim(),
    imageInputName: els.imageInput.value.trim() || "image",
    streamOutputNames: (els.streamOutputs.value || "output_image")
      .split(",").map(s => s.trim()).filter(Boolean),
    dataOutputNames: (els.dataOutputs.value || "predictions")
      .split(",").map(s => s.trim()).filter(Boolean),
    requestedRegion: els.region.value || "us",
    requestedPlan: els.plan.value || "webrtc-gpu-small",
    processingTimeout:
      Number.parseInt(els.timeout.value, 10) || 600,
    confidence: Number(els.threshold.value) / 100
  };
}

function savePipeline() {
  saveJSON(STORAGE.pipeline, readPipeline());
  updateThresholdDisplay();
  updateModeHelp();
  setStatus("Pipeline configuration saved in this browser.");
}

function restorePipeline() {
  const config = loadJSON(STORAGE.pipeline, null);
  if (!config) return;

  if (config.mode) els.inferenceMode.value = config.mode;
  if (config.modelId) els.modelId.value = config.modelId;
  if (config.workspaceName) els.workspace.value = config.workspaceName;
  if (config.workflowId) els.workflowId.value = config.workflowId;
  if (config.imageInputName) els.imageInput.value = config.imageInputName;
  if (config.streamOutputNames?.length) {
    els.streamOutputs.value = config.streamOutputNames.join(",");
  }
  if (config.dataOutputNames?.length) {
    els.dataOutputs.value = config.dataOutputNames.join(",");
  }
  if (config.requestedRegion) els.region.value = config.requestedRegion;
  if (config.requestedPlan) els.plan.value = config.requestedPlan;
  if (config.processingTimeout) {
    els.timeout.value = config.processingTimeout;
  }
  if (Number.isFinite(config.confidence)) {
    els.threshold.value = String(Math.round(config.confidence * 100));
  }
}

function updateThresholdDisplay() {
  const percent = `${els.threshold.value}%`;
  els.thresholdLabel.textContent = percent;
  els.thresholdMetric.textContent = percent;
  els.dashThreshold.textContent = percent;
}

function updateModeHelp() {
  const dynamic = els.inferenceMode.value === "dynamic";

  els.modelIdField.classList.toggle("hidden", !dynamic);

  els.thresholdHelp.textContent = dynamic
    ? "Applied inside the model block in the generated workflow. Apply changes to restart the stream with the new threshold."
    : "Saved Workflow mode uses the confidence settings saved in Roboflow. This slider cannot change that workflow's actual threshold.";
}

els.inferenceMode.addEventListener("change", updateModeHelp);
els.threshold.addEventListener("input", updateThresholdDisplay);

$("savePipelineBtn").addEventListener("click", savePipeline);

$("applyPipelineBtn").addEventListener("click", async () => {
  savePipeline();

  if (connection) {
    reconnecting = true;
    await stopStream();
    reconnecting = false;
    await startStream();
  } else {
    setStatus("Configuration saved. Start the stream to apply it.");
  }
});

// ============================================================
// REAL SERVER-SIDE CONFIDENCE THRESHOLD
// ============================================================

/**
 * Dynamic Model mode creates a workflow whose model block receives
 * the threshold selected in the dashboard.
 *
 * The threshold is sent as workflow configuration to the inference
 * backend. It is not applied as a frontend-only display filter.
 */
function buildDynamicWorkflow(modelId, confidence, workspaceName) {
  if (!modelId) {
    throw new Error(
      "Enter your exact trained Roboflow model ID in Pipeline settings."
    );
  }

  const fullModelId = modelId.includes("/") ? modelId : `${workspaceName}/${modelId}`;
  if (!workspaceName && !modelId.includes("/")) {
    throw new Error("Enter your Roboflow workspace name in Pipeline settings.");
  }

  return {
    version: "1.0",

    inputs: [
      {
        type: "WorkflowImage",
        name: "image"
      }
    ],

    steps: [
      {
        type: "roboflow_core/roboflow_object_detection_model@v3",
        name: "model_1",
        images: "$inputs.image",
        model_id: fullModelId,
        confidence_mode: "custom",
        custom_confidence: confidence
      },
      {
        type: "roboflow_core/bounding_box_visualization@v1",
        name: "bounding_box_visualization",
        image: "$inputs.image",
        predictions: "$steps.model_1.predictions"
      },
      {
        type: "roboflow_core/label_visualization@v2",
        name: "label_visualization",
        image: "$steps.bounding_box_visualization.image",
        predictions: "$steps.model_1.predictions",
        text: "Class"
      }
    ],

    outputs: [
      {
        type: "JsonField",
        name: "output_image",
        coordinates_system: "own",
        selector: "$steps.label_visualization.image"
      },
      {
        type: "JsonField",
        name: "predictions",
        coordinates_system: "own",
        selector: "$steps.model_1.predictions"
      }
    ]
  };
}

// ============================================================
// CAMERA AND ROBoflow WEBRTC
// ============================================================

function getCameraConfig() {
  const resolution = $("resolutionSelect")?.value?.split("x") || [];

  return {
    deviceId: $("cameraSelect")?.value || undefined,
    width: Number.parseInt(resolution[0], 10) || 640,
    height: Number.parseInt(resolution[1], 10) || 480,
    frameRate: Number.parseInt($("fpsSelect")?.value, 10) || 15
  };
}

async function connectToRoboflow() {
  const config = readPipeline();
  const camera = getCameraConfig();

  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error(
      "Camera access is unavailable. Open the site over HTTPS."
    );
  }

  let wrtcParams;

  if (config.mode === "dynamic") {
    const workflowSpec = buildDynamicWorkflow(
      config.modelId,
      config.confidence,
      config.workspaceName
    );

    wrtcParams = {
      workflowSpec,
      imageInputName: "image",
      streamOutputNames: ["output_image"],
      dataOutputNames: ["predictions"],
      requestedRegion: config.requestedRegion,
      requestedPlan: config.requestedPlan,
      processingTimeout: config.processingTimeout
    };
  } else {
    if (!config.workspaceName || !config.workflowId) {
      throw new Error(
        "Enter both workspace name and saved workflow ID."
      );
    }

    wrtcParams = {
      workspaceName: config.workspaceName,
      workflowId: config.workflowId,
      imageInputName: config.imageInputName,
      streamOutputNames: config.streamOutputNames,
      dataOutputNames: config.dataOutputNames,
      requestedRegion: config.requestedRegion,
      requestedPlan: config.requestedPlan,
      processingTimeout: config.processingTimeout
    };
  }

  const videoConstraints = {
    width: { ideal: camera.width },
    height: { ideal: camera.height },
    frameRate: { ideal: camera.frameRate, max: camera.frameRate }
  };

  if (camera.deviceId) {
    videoConstraints.deviceId = { exact: camera.deviceId };
  }

  const connector = connectors.withProxyUrl(
    "/api/init-webrtc",
    { turnConfigUrl: "/api/turn-config" }
  );

  const source = await streams.useCamera({
    video: videoConstraints,
    audio: false
  });

  try {
    return await webrtc.useStream({
      source,
      connector,
      wrtcParams,
      onData: handleInferenceData,
      options: {
        disableInputStreamDownscaling: false
      }
    });
  } catch (error) {
    source.getTracks?.().forEach(track => track.stop());
    throw error;
  }
}

// ============================================================
// START AND STOP
// ============================================================

async function startStream() {
  if (starting || connection) return;

  starting = true;
  setButtonState();
  setStatus("Connecting camera and inference workflow...", "connecting");
  els.feedHint.textContent = "Waiting for camera and Roboflow connection...";
  els.placeholder.classList.remove("hidden");

  try {
    const newConnection = await connectToRoboflow();
    connection = newConnection;

    const remoteStream = await newConnection.remoteStream();

    if (!remoteStream) {
      throw new Error("Roboflow did not return a processed video stream.");
    }

    els.video.srcObject = remoteStream;
    els.video.muted = true;
    els.video.autoplay = true;
    els.video.playsInline = true;

    try {
      await els.video.play();
    } catch (error) {
      console.warn("[Video] Autoplay could not start:", error);
    }

    els.placeholder.classList.add("hidden");
    els.feedHint.textContent = "Live processed video from Roboflow.";
    setStatus("Live — inference is running.", "live");
    els.feedState.textContent = "LIVE";

  } catch (error) {
    console.error("[VISION LAB] Start failed:", error);

    if (connection) {
      try { await connection.cleanup(); } catch {}
      connection = null;
    }

    els.video.srcObject = null;
    els.placeholder.classList.remove("hidden");

    setStatus(`Connection failed: ${error.message || "Unknown error"}`, "error");
    els.feedHint.textContent =
      "Check camera permissions, model/workflow settings and Render logs.";
  } finally {
    starting = false;
    setButtonState();
  }
}

async function stopStream() {
  if (!connection || stopping) return;

  stopping = true;
  const current = connection;
  connection = null;
  setButtonState();
  setStatus("Stopping stream...");

  try {
    await current.cleanup();
  } catch (error) {
    console.warn("[VISION LAB] Cleanup:", error);
  }

  if (els.video.srcObject) {
    els.video.pause();
    els.video.srcObject = null;
  }

  els.placeholder.classList.remove("hidden");
  els.feedHint.textContent = "Camera is not connected.";

  stopping = false;
  setButtonState();

  if (!reconnecting) {
    setStatus("Idle");
  }
}

els.start.addEventListener("click", startStream);
els.stop.addEventListener("click", stopStream);

// ============================================================
// NORMALIZE INFERENCE RESULTS
// ============================================================

function isPrediction(value) {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const name = value.class ?? value.class_name ?? value.className ?? value.label ?? value.category;
  return typeof name === "string" && name.trim().length > 0;
}

function findPredictions(data, depth = 0) {
  if (depth > 8 || data == null) return [];

  if (typeof data === "string") {
    const trimmed = data.trim();
    if ((trimmed.startsWith("{") && trimmed.endsWith("}")) ||
        (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
      try { return findPredictions(JSON.parse(trimmed), depth + 1); }
      catch { return []; }
    }
    return [];
  }

  if (Array.isArray(data)) {
    const direct = data.filter(isPrediction);
    if (direct.length) return direct;
    for (const item of data) {
      const nested = findPredictions(item, depth + 1);
      if (nested.length) return nested;
    }
    return [];
  }

  if (typeof data !== "object") return [];

  // Common direct prediction arrays from object-detection workflows.
  for (const key of ["predictions", "detections", "objects", "results"]) {
    if (Array.isArray(data[key])) {
      const direct = data[key].filter(isPrediction);
      if (direct.length) return direct;
      for (const item of data[key]) {
        const nested = findPredictions(item, depth + 1);
        if (nested.length) return nested;
      }
    }
  }

  // Search nested workflow outputs, skipping image/display payloads.
  for (const [key, value] of Object.entries(data)) {
    if (/^(image|frame|video|canvas|visualization|metadata|time|timestamp)$/i.test(key)) continue;
    const nested = findPredictions(value, depth + 1);
    if (nested.length) return nested;
  }

  return [];
}

function predictionName(prediction) {
  const value = prediction?.class ?? prediction?.class_name ??
    prediction?.className ?? prediction?.label ?? prediction?.category;
  return typeof value === "string" && value.trim() ? value.trim() : "Unknown object";
}

function confidencePercent(prediction) {
  const raw = prediction?.confidence ?? prediction?.class_confidence ??
    prediction?.classConfidence ?? prediction?.score;
  if (raw === null || raw === undefined || raw === "") return null;

  const number = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(number)) return null;

  // Roboflow scores are commonly 0–1, but some output formats use 0–100.
  const percent = number <= 1 ? number * 100 : number;
  return Math.round(Math.max(0, Math.min(100, percent)));
}

function handleInferenceData(data) {
  console.log("[VISION LAB] Raw inference data:", data);

  const predictions = findPredictions(data);
  console.log("[VISION LAB] Parsed predictions:", predictions);

  lastDetections = predictions;
  eventCount++;
  predictions.forEach(prediction => sessionClasses.add(predictionName(prediction)));

  if (els.totalDetections) els.totalDetections.textContent = String(eventCount);
  if (els.uniqueClasses) els.uniqueClasses.textContent = String(sessionClasses.size);
  if (els.dashEvents) els.dashEvents.textContent = String(eventCount);
  if (els.dashClasses) els.dashClasses.textContent = String(sessionClasses.size);

  renderDetections(predictions);
  renderDashboardRecent();
  if (predictions.length) updateHistoryFromPredictions(predictions);
}

// ============================================================
// LIVE DETECTION SIDEBAR
// ============================================================
// ============================================================
// LIVE DETECTION SIDEBAR
// ============================================================

function renderDetections(predictions) {
  if (els.visibleCount) {
    els.visibleCount.textContent =
      `${predictions.length} item${predictions.length === 1 ? "" : "s"}`;
  }

  if (!predictions.length) {
    els.detectionList.innerHTML =
      '<div class="empty-state">No objects returned for this frame.<br/>Try improving lighting or lowering the pipeline threshold.</div>';
    return;
  }

  els.detectionList.replaceChildren();

  predictions.forEach(prediction => {
    const name = predictionName(prediction);
    const confidence = confidencePercent(prediction);

    const item = document.createElement("div");
    item.className = "detection-item";

    const top = document.createElement("div");
    top.className = "detection-top";

    const label = document.createElement("span");
    label.className = "detection-name";
    label.textContent = name;

    const score = document.createElement("span");
    score.className = "confidence";
    score.textContent =
      confidence === null ? "Score unavailable" : `${confidence}%`;

    top.append(label, score);
    item.appendChild(top);

    if (confidence !== null) {
      const meter = document.createElement("div");
      meter.className = "meter";

      const bar = document.createElement("span");
      bar.style.width = `${confidence}%`;

      meter.appendChild(bar);
      item.appendChild(meter);
    }

    els.detectionList.appendChild(item);
  });
}

// ============================================================
// READING HISTORY
// ============================================================

function updateHistoryFromPredictions(predictions) {
  if (!predictions.length) return;

  const now = new Date().toISOString();

  predictions.forEach(prediction => {
    history.unshift({
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      time: now,
      name: predictionName(prediction),
      confidence: confidencePercent(prediction)
    });
  });

  // Keep browser storage bounded.
  history = history.slice(0, 500);
  saveJSON(STORAGE.history, history);
  renderHistory();
}

function formatTime(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? "Unknown time"
    : date.toLocaleString();
}

function renderHistory() {
  if (!els.historyList) return;
  const query = (els.historySearch?.value || "").trim().toLowerCase();

  const filtered = history.filter(item =>
    item.name.toLowerCase().includes(query)
  );

  els.historyList.replaceChildren();

  if (!filtered.length) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = history.length
      ? "No history matches your search."
      : "No detections have been recorded yet.";
    els.historyList.appendChild(empty);
    return;
  }

  filtered.forEach(item => {
    const row = document.createElement("div");
    row.className = "history-row";

    const left = document.createElement("div");
    const title = document.createElement("div");
    title.className = "history-title";
    title.textContent = item.name;

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = formatTime(item.time);

    left.append(title, meta);

    const score = document.createElement("div");
    score.className = "history-confidence";
    score.textContent =
      item.confidence === null ? "N/A" : `${item.confidence}%`;

    row.append(left, score);
    els.historyList.appendChild(row);
  });
}

els.historySearch?.addEventListener("input", renderHistory);

$("clearHistoryBtn").addEventListener("click", () => {
  if (!confirm("Clear all saved detection history from this browser?")) {
    return;
  }

  history = [];
  saveJSON(STORAGE.history, history);
  renderHistory();
});

// ============================================================
// DASHBOARD SUMMARY
// ============================================================

function renderDashboardRecent() {
  if (!els.dashboardRecent) return;
  const recent = history.slice(0, 6);
  els.dashboardRecent.replaceChildren();

  if (!recent.length) {
    els.dashboardRecent.textContent =
      "No detection events recorded yet.";
    return;
  }

  recent.forEach(item => {
    const row = document.createElement("div");
    row.className = "history-row";

    const left = document.createElement("div");
    const title = document.createElement("div");
    title.className = "history-title";
    title.textContent = item.name;

    const meta = document.createElement("div");
    meta.className = "history-meta";
    meta.textContent = formatTime(item.time);

    left.append(title, meta);

    const score = document.createElement("div");
    score.className = "history-confidence";
    score.textContent =
      item.confidence === null ? "N/A" : `${item.confidence}%`;

    row.append(left, score);
    els.dashboardRecent.appendChild(row);
  });
}

// ============================================================
// THEME AND DISPLAY SETTINGS
// ============================================================

const themes = {
  cerulean:        { accent: "#00bfff", rgb: "0,191,255" },
  "neon-rose":     { accent: "#ff2d75", rgb: "255,45,117" },
  "laser-purple":  { accent: "#b026ff", rgb: "176,38,255" },
  "acid-green":    { accent: "#b6ff00", rgb: "182,255,0" },
  "electric-blue": { accent: "#3d5afe", rgb: "61,90,254" },
  "neon-orange":   { accent: "#ff6d00", rgb: "255,109,0" },
  "ice-cyan":      { accent: "#48f5ff", rgb: "72,245,255" },
  "hot-magenta":   { accent: "#ff00aa", rgb: "255,0,170" }
};

function applyTheme(name) {
  const theme = themes[name] || themes.cerulean;
  const root = document.documentElement;

  // Keep the jet-black surfaces fixed and drive all neon accents from the selected theme.
  root.style.setProperty("--accent", theme.accent);
  root.style.setProperty("--accent-rgb", theme.rgb);
  root.style.setProperty("--rose", theme.accent);
  root.style.setProperty("--rose-rgb", theme.rgb);
  root.style.setProperty("--bg", "#030405");
  root.style.setProperty("--panel", "#0a0c10");
  root.style.setProperty("--panel2", "#10141b");
  root.style.setProperty("--border", "rgba(" + theme.rgb + ", .32)");
  root.style.setProperty("--danger", theme.accent);
  root.style.setProperty("--purple", theme.accent);
  root.style.setProperty("--green", theme.accent);

  document.querySelectorAll(".theme-btn").forEach(button => {
    button.classList.toggle("active", button.dataset.theme === name);
  });

  const settings = loadJSON(STORAGE.settings, {});
  settings.theme = name;
  saveJSON(STORAGE.settings, settings);
}

document.querySelectorAll(".theme-btn").forEach(button => {
  button.addEventListener("click", () => applyTheme(button.dataset.theme));
});

function saveSettings() {
  const settings = {
    ...loadJSON(STORAGE.settings, {}),
    theme:
      document.querySelector(".theme-btn.active")?.dataset.theme || "cerulean",
    density: els.density.value,
    boxStyle: els.boxStyle.value,
    showSidebar: els.showSidebar.checked
  };

  saveJSON(STORAGE.settings, settings);
  applyDisplaySettings(settings);
  setStatus("Appearance settings saved.");
}

function applyDisplaySettings(settings) {
  document.body.classList.toggle(
    "compact",
    settings.density === "compact"
  );

  // Compact mode reduces spacing without affecting the camera stream.
  document.documentElement.style.setProperty(
    "--radius",
    settings.density === "compact" ? "10px" : "16px"
  );

  const sidebarPanel = els.detectionList.closest("aside");
  if (sidebarPanel) {
    sidebarPanel.classList.toggle(
      "hidden",
      settings.showSidebar === false
    );
  }

  els.density.value = settings.density || "comfortable";
  els.boxStyle.value = settings.boxStyle || "normal";
  els.showSidebar.checked = settings.showSidebar !== false;
}

$("saveSettingsBtn").addEventListener("click", saveSettings);

function restoreSettings() {
  const settings = loadJSON(STORAGE.settings, {});

  applyTheme(settings.theme || "cerulean");
  applyDisplaySettings(settings);
}

// ============================================================
// BACKEND HEALTH — NO SECRET VALUES ARE DISPLAYED
// ============================================================

async function checkBackendHealth() {
  els.backendState.textContent = "Checking";
  els.apiKeyState.textContent = "Checking";

  try {
    const response = await fetch("/api/health", {
      cache: "no-store"
    });

    if (!response.ok) {
      throw new Error(`Health endpoint returned HTTP ${response.status}`);
    }

    const data = await response.json();
    lastHealth = data;

    els.backendState.textContent = "Online";
    els.apiKeyState.textContent =
      data.apiKeyConfigured === true ? "Configured" :
      data.apiKeyConfigured === false ? "Missing" : "Unknown";

    console.log("[Backend health]", {
      online: true,
      apiKeyConfigured: data.apiKeyConfigured
    });
  } catch (error) {
    lastHealth = null;
    els.backendState.textContent = "Unavailable";
    els.apiKeyState.textContent = "Not verified";
    console.warn("[Backend health]", error);
  }
}

$("healthCheckBtn").addEventListener("click", checkBackendHealth);

// ============================================================
// CAMERA SETTINGS — OPTIONAL CONTROLS IN THE PAGE
// ============================================================

async function initializeCameraList() {
  const select = $("cameraSelect");
  if (!select || !navigator.mediaDevices?.enumerateDevices) return;

  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(device => device.kind === "videoinput");

    select.replaceChildren();

    if (!cameras.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Default camera";
      select.appendChild(option);
      return;
    }

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");
      option.value = camera.deviceId;
      option.textContent = camera.label || `Camera ${index + 1}`;
      select.appendChild(option);
    });
  } catch (error) {
    console.warn("[Camera list]", error);
  }
}

// ============================================================
// INITIALIZATION
// ============================================================

restorePipeline();
restoreSettings();
updateThresholdDisplay();
updateModeHelp();
renderHistory();
renderDashboardRecent();
setButtonState();
setStatus("Ready — click Start.");
checkBackendHealth();
initializeCameraList();

window.addEventListener("pagehide", () => {
  if (connection) {
    connection.cleanup().catch(error => {
      console.warn("[Page cleanup]", error);
    });
  }
});
