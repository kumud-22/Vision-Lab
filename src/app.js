
/**
 * VISION LAB — Roboflow Secure WebRTC Object Detector
 *
 * Improvements:
 * - No camera access request just to load the page.
 * - Camera is requested only when Start is clicked.
 * - Lower default camera resolution and FPS for faster streaming.
 * - Secure backend proxy keeps the Roboflow API key server-side.
 * - Better connection status, error handling, and cleanup.
 *
 * IMPORTANT:
 * Keep your existing backend and Render environment variables.
 */

import { connectors, webrtc, streams } from "@roboflow/inference-sdk";

// ============================================================
// 1. DOM ELEMENTS
// ============================================================

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusEl = document.getElementById("status");
const videoEl = document.getElementById("video");
const dataPreviewEl = document.getElementById("dataPreview");
const dataCountEl = document.getElementById("dataCount");

const configInputs = {
  workspaceName: document.getElementById("workspaceName"),
  workflowId: document.getElementById("workflowId"),
  imageInputName: document.getElementById("imageInputName"),
  streamOutputNames: document.getElementById("streamOutputNames"),
  dataOutputNames: document.getElementById("dataOutputNames")
};

const exampleInputs = {
  streamOutput: document.getElementById("exampleStreamOutput")
};

const serverInputs = {
  requestedRegion: document.getElementById("requestedRegion"),
  requestedPlan: document.getElementById("requestedPlan"),
  processingTimeout: document.getElementById("processingTimeout")
};

const cameraInputs = {
  cameraSelect: document.getElementById("cameraSelect"),
  resolutionSelect: document.getElementById("resolutionSelect"),
  fpsSelect: document.getElementById("fpsSelect"),
  cameraCaps: document.getElementById("cameraCaps")
};

const tabBtns = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

// ============================================================
// 2. STATE
// ============================================================

let workflowMode = "example";
let activeConnection = null;
let dataMessageCount = 0;
let isStarting = false;
let isStopping = false;

// ============================================================
// 3. STATUS AND UI HELPERS
// ============================================================

function setStatus(message) {
  if (statusEl) {
    statusEl.textContent = message;
  }

  console.log("[VISION LAB]", message);
}

function setButtons({ starting = false, connected = false } = {}) {
  if (startBtn) {
    startBtn.disabled = starting || connected;
  }

  if (stopBtn) {
    stopBtn.disabled = !connected;
  }
}

function resetDataDisplay() {
  dataMessageCount = 0;

  if (dataCountEl) {
    dataCountEl.textContent = "0";
  }

  if (dataPreviewEl) {
    dataPreviewEl.textContent = "";
  }
}

function showError(message) {
  console.error("[VISION LAB] Error:", message);
  setStatus(`Error: ${message}`);
}

// ============================================================
// 4. SERVER CONFIGURATION
// ============================================================

function getServerConfig() {
  return {
    requestedRegion: serverInputs.requestedRegion?.value || "us",
    requestedPlan:
      serverInputs.requestedPlan?.value || "webrtc-gpu-small",
    processingTimeout:
      Number.parseInt(serverInputs.processingTimeout?.value, 10) || 600
  };
}

// ============================================================
// 5. WORKFLOW CONFIGURATION
// ============================================================

function getConfig() {
  const serverConfig = getServerConfig();

  if (workflowMode === "example") {
    const streamOutput =
      exampleInputs.streamOutput?.value || "labels";

    return {
      mode: "example",
      workflowSpec: WORKFLOW_SPEC,
      imageInputName: "image",
      streamOutputNames: [streamOutput],
      dataOutputNames: ["count"],
      ...serverConfig
    };
  }

  const workspaceName =
    configInputs.workspaceName?.value?.trim();

  const workflowId =
    configInputs.workflowId?.value?.trim();

  if (!workspaceName || !workflowId) {
    throw new Error(
      "Open the Custom tab and enter your Roboflow workspace and workflow ID."
    );
  }

  return {
    mode: "custom",
    workspaceName,
    workflowId,

    imageInputName:
      configInputs.imageInputName?.value?.trim() || "image",

    streamOutputNames: (
      configInputs.streamOutputNames?.value?.trim() ||
      "output_image"
    )
      .split(",")
      .map(value => value.trim())
      .filter(Boolean),

    dataOutputNames: (
      configInputs.dataOutputNames?.value?.trim() ||
      "predictions"
    )
      .split(",")
      .map(value => value.trim())
      .filter(Boolean),

    ...serverConfig
  };
}

// ============================================================
// 6. TAB SWITCHING
// ============================================================

function switchTab(tabName) {
  if (tabName !== "example" && tabName !== "custom") {
    return;
  }

  workflowMode = tabName;

  tabBtns.forEach(button => {
    button.classList.toggle(
      "active",
      button.dataset.tab === tabName
    );
  });

  tabContents.forEach(content => {
    content.classList.toggle(
      "active",
      content.id === `tab-${tabName}`
    );
  });

  console.log("[Workflow] Selected mode:", tabName);
}

tabBtns.forEach(button => {
  button.addEventListener("click", () => {
    switchTab(button.dataset.tab);
  });
});

// ============================================================
// 7. CAMERA SETTINGS
// ============================================================

function getCameraConfig() {
  const resolution =
    cameraInputs.resolutionSelect?.value?.split("x") || [];

  const width = Number.parseInt(resolution[0], 10) || 640;
  const height = Number.parseInt(resolution[1], 10) || 480;
  const frameRate =
    Number.parseInt(cameraInputs.fpsSelect?.value, 10) || 15;

  return {
    deviceId: cameraInputs.cameraSelect?.value || undefined,
    width,
    height,
    frameRate
  };
}

/**
 * Set performance-friendly defaults.
 * This function does not open the camera.
 */
function setDefaultCameraOptions() {
  if (cameraInputs.resolutionSelect) {
    cameraInputs.resolutionSelect.innerHTML = `
      <option value="640x480" selected>VGA (640×480)</option>
      <option value="1280x720">720p (1280×720)</option>
      <option value="1920x1080">1080p (1920×1080)</option>
    `;

    cameraInputs.resolutionSelect.value = "640x480";
    cameraInputs.resolutionSelect.disabled = false;
  }

  if (cameraInputs.fpsSelect) {
    cameraInputs.fpsSelect.innerHTML = `
      <option value="15" selected>15 fps</option>
      <option value="24">24 fps</option>
      <option value="30">30 fps</option>
    `;

    cameraInputs.fpsSelect.value = "15";
    cameraInputs.fpsSelect.disabled = false;
  }
}

/**
 * List available cameras without opening a camera stream.
 */
async function enumerateCameras() {
  const cameraSelect = cameraInputs.cameraSelect;

  setDefaultCameraOptions();

  if (!cameraSelect) {
    return;
  }

  if (!navigator.mediaDevices?.enumerateDevices) {
    cameraSelect.innerHTML =
      '<option value="">Camera access unavailable</option>';

    if (cameraInputs.cameraCaps) {
      cameraInputs.cameraCaps.textContent =
        "Use a supported browser over HTTPS.";
    }

    return;
  }

  try {
    const devices =
      await navigator.mediaDevices.enumerateDevices();

    const cameras = devices.filter(
      device => device.kind === "videoinput"
    );

    cameraSelect.innerHTML = "";

    if (cameras.length === 0) {
      cameraSelect.innerHTML =
        '<option value="">Default camera</option>';

      if (cameraInputs.cameraCaps) {
        cameraInputs.cameraCaps.textContent =
          "No camera listed yet. Click Start and allow camera access.";
      }

      return;
    }

    cameras.forEach((camera, index) => {
      const option = document.createElement("option");

      option.value = camera.deviceId;
      option.textContent =
        camera.label || `Camera ${index + 1}`;

      cameraSelect.appendChild(option);
    });

    cameraSelect.disabled = false;

    if (cameraInputs.cameraCaps) {
      cameraInputs.cameraCaps.textContent =
        "Camera list loaded. Camera access begins when you click Start.";
    }

    console.log("[Camera] Available cameras:", cameras);
  } catch (error) {
    console.warn("[Camera] Could not list cameras:", error);

    cameraSelect.innerHTML =
      '<option value="">Use default camera</option>';

    if (cameraInputs.cameraCaps) {
      cameraInputs.cameraCaps.textContent =
        "The browser will try the default camera.";
    }
  }
}

// Selecting a camera no longer opens it just to inspect capabilities.
cameraInputs.cameraSelect?.addEventListener("change", () => {
  if (cameraInputs.cameraCaps) {
    cameraInputs.cameraCaps.textContent =
      "Selected camera. It will open when you click Start.";
  }
});

// ============================================================
// 8. EXAMPLE WORKFLOW
// ============================================================

const WORKFLOW_SPEC = {
  version: "1.0",

  inputs: [
    {
      type: "InferenceImage",
      name: "image"
    }
  ],

  steps: [
    {
      type: "roboflow_core/roboflow_object_detection_model@v2",
      name: "model",
      images: "$inputs.image",
      model_id: "rfdetr-nano"
    },
    {
      type: "roboflow_core/blur_visualization@v1",
      name: "blur_visualization",
      image: "$inputs.image",
      predictions: "$steps.model.predictions"
    },
    {
      type: "roboflow_core/bounding_box_visualization@v1",
      name: "bounding_box_visualization",
      image: "$inputs.image",
      predictions: "$steps.model.predictions"
    },
    {
      type: "roboflow_core/label_visualization@v1",
      name: "label_visualization",
      image: "$steps.bounding_box_visualization.image",
      predictions: "$steps.model.predictions"
    },
    {
      type: "roboflow_core/property_definition@v1",
      name: "property_definition",
      data: "$steps.model.predictions",
      operations: [
        {
          type: "SequenceLength"
        }
      ]
    }
  ],

  outputs: [
    {
      type: "JsonField",
      name: "blur",
      coordinates_system: "own",
      selector: "$steps.blur_visualization.image"
    },
    {
      type: "JsonField",
      name: "labels",
      coordinates_system: "own",
      selector: "$steps.label_visualization.image"
    },
    {
      type: "JsonField",
      name: "count",
      coordinates_system: "own",
      selector: "$steps.property_definition.output"
    }
  ]
};

// ============================================================
// 9. SECURE ROBoflow WEBRTC CONNECTION
// ============================================================

async function connectWebcamToRoboflowWebRTC() {
  const config = getConfig();
  const cameraConfig = getCameraConfig();

  console.log("[Config] Workflow mode:", config.mode);
  console.log("[Config] Camera:", cameraConfig);

  // API key remains on the backend.
  const connector = connectors.withProxyUrl(
    "/api/init-webrtc",
    {
      turnConfigUrl: "/api/turn-config"
    }
  );

  const videoConstraints = {
    width: {
      ideal: cameraConfig.width
    },
    height: {
      ideal: cameraConfig.height
    },
    frameRate: {
      ideal: cameraConfig.frameRate,
      max: cameraConfig.frameRate
    }
  };

  if (cameraConfig.deviceId) {
    videoConstraints.deviceId = {
      exact: cameraConfig.deviceId
    };
  } else {
    videoConstraints.facingMode = {
      ideal: "environment"
    };
  }

  const baseParams = {
    imageInputName: config.imageInputName,
    streamOutputNames: config.streamOutputNames,
    dataOutputNames: config.dataOutputNames,
    requestedRegion: config.requestedRegion,
    requestedPlan: config.requestedPlan,
    processingTimeout: config.processingTimeout
  };

  const wrtcParams =
    config.mode === "example"
      ? {
          ...baseParams,
          workflowSpec: config.workflowSpec
        }
      : {
          ...baseParams,
          workspaceName: config.workspaceName,
          workflowId: config.workflowId
        };

  // Allow the SDK to downscale the outgoing video if needed.
  // This can reduce bandwidth and processing load.
  const connection = await webrtc.useStream({
    source: await streams.useCamera({
      video: videoConstraints,
      audio: false
    }),

    connector,
    wrtcParams,

    onData: handleData,

    options: {
      disableInputStreamDownscaling: false
    }
  });

  return connection;
}

// ============================================================
// 10. INCOMING DETECTION DATA
// ============================================================

function handleData(data) {
  console.log("[Detection data]", data);

  dataMessageCount++;

  if (dataCountEl) {
    dataCountEl.textContent = String(dataMessageCount);
  }

  if (dataPreviewEl) {
    try {
      dataPreviewEl.textContent = JSON.stringify(data, null, 2);
    } catch {
      dataPreviewEl.textContent = String(data);
    }
  }
}

// ============================================================
// 11. START STREAMING
// ============================================================

async function start() {
  if (isStarting || activeConnection) {
    return;
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    showError(
      "Camera access is unavailable. Open the website over HTTPS and use a supported browser."
    );
    return;
  }

  isStarting = true;
  setButtons({ starting: true });
  resetDataDisplay();

  setStatus("Preparing camera...");

  let newConnection = null;

  try {
    // This SDK call obtains camera access and establishes the stream.
    setStatus("Requesting camera and connecting to Roboflow...");

    newConnection = await connectWebcamToRoboflowWebRTC();

    activeConnection = newConnection;

    setStatus("Establishing video stream...");

    const remoteStream = await newConnection.remoteStream();

    if (!remoteStream) {
      throw new Error(
        "The connection opened but no remote video stream was returned."
      );
    }

    videoEl.srcObject = remoteStream;
    videoEl.controls = false;
    videoEl.muted = true;
    videoEl.autoplay = true;
    videoEl.playsInline = true;

    try {
      await videoEl.play();
    } catch (playError) {
      console.warn(
        "[Video] Autoplay was blocked:",
        playError
      );

      setStatus(
        "Connected. If video stays paused, press Play on the video."
      );
    }

    // Permission is now likely available, so camera names may be clearer.
    enumerateCameras().catch(error => {
      console.warn("[Camera] Refresh failed:", error);
    });

    setButtons({ connected: true });
    setStatus("Connected — AI is processing video.");

    console.log("[VISION LAB] Streaming started.");
  } catch (error) {
    console.error("[VISION LAB] Startup failed:", error);

    if (newConnection) {
      try {
        await newConnection.cleanup();
      } catch (cleanupError) {
        console.warn(
          "[Cleanup] Failed to clean up failed connection:",
          cleanupError
        );
      }
    }

    if (activeConnection === newConnection) {
      activeConnection = null;
    }

    if (videoEl) {
      videoEl.srcObject = null;
    }

    const message =
      error?.message || "An unknown connection error occurred.";

    if (/permission|denied|notallowed/i.test(message)) {
      showError(
        "Camera permission was denied. Allow camera access in your browser settings, then try again."
      );
    } else if (/notfound|device not found/i.test(message)) {
      showError(
        "No usable camera was found. Check that your camera is connected and not being used exclusively by another app."
      );
    } else if (/overconstrained/i.test(message)) {
      showError(
        "The selected camera does not support these settings. Choose a lower resolution or use the default camera."
      );
    } else if (/api key/i.test(message)) {
      showError(
        "The server API key is not configured correctly. Check the ROBOFLOW_API_KEY environment variable in Render."
      );
    } else {
      showError(message);
    }

    setButtons();
  } finally {
    isStarting = false;

    if (!activeConnection && startBtn) {
      startBtn.disabled = false;
    }
  }
}

// ============================================================
// 12. STOP STREAMING
// ============================================================

async function stop() {
  if (!activeConnection || isStopping) {
    return;
  }

  isStopping = true;

  const connectionToStop = activeConnection;

  if (stopBtn) {
    stopBtn.disabled = true;
  }

  setStatus("Stopping stream...");

  try {
    await connectionToStop.cleanup();
    console.log("[VISION LAB] Connection cleaned up.");
  } catch (error) {
    console.error("[Cleanup] Error stopping connection:", error);
  } finally {
    if (activeConnection === connectionToStop) {
      activeConnection = null;
    }

    if (videoEl) {
      videoEl.pause();
      videoEl.srcObject = null;
    }

    resetDataDisplay();
    setButtons();
    setStatus("Idle");

    isStopping = false;
  }
}

// ============================================================
// 13. BUTTON LISTENERS
// ============================================================

startBtn?.addEventListener("click", start);
stopBtn?.addEventListener("click", stop);

// ============================================================
// 14. SERVER HEALTH CHECK
// ============================================================

async function checkServerHealth() {
  try {
    const response = await fetch("/api/health");

    if (!response.ok) {
      throw new Error(`Health check returned HTTP ${response.status}`);
    }

    const data = await response.json();

    console.log("[Server health]", data);

    if (data.apiKeyConfigured === false) {
      console.warn(
        "[Server] ROBOFLOW_API_KEY is not configured."
      );
    }
  } catch (error) {
    // A failed health check should not prevent the user from trying Start.
    console.warn("[Server] Health check failed:", error);
  }
}

// ============================================================
// 15. PAGE INITIALIZATION AND CLEANUP
// ============================================================

setButtons();
setStatus("Ready — click Start to begin.");

enumerateCameras();
checkServerHealth();

async function cleanupOnExit() {
  if (activeConnection) {
    const connection = activeConnection;
    activeConnection = null;

    try {
      await connection.cleanup();
    } catch (error) {
      console.warn("[Cleanup] Exit cleanup failed:", error);
    }
  }
}

window.addEventListener("pagehide", cleanupOnExit);
