/* TrafficLens mobile client: camera -> WebSocket -> YOLO on the GPU -> overlay. */
(() => {
  "use strict";

  const MAX_SIDE = 640;
  const SEND_INTERVAL_MS = 40;
  const JPEG_QUALITY = 0.72;
  const CLASS_KEYS = ["car", "truck", "bus", "motorcycle"];
  const CLASS_COLORS = {
    car: "#22d3ee",
    truck: "#fb923c",
    bus: "#a78bfa",
    motorcycle: "#f472b6",
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    video: $("video"),
    overlay: $("overlay"),
    stage: $("stage"),
    viewer: $("viewer"),
    placeholderTitle: $("placeholderTitle"),
    placeholderText: $("placeholderText"),
    stageAlert: $("stageAlert"),
    statusPill: $("statusPill"),
    statusText: $("statusText"),
    backendBadge: $("backendBadge"),
    fpsBadge: $("fpsBadge"),
    startBtn: $("startBtn"),
    startLabel: $("startLabel"),
    screenBtn: $("screenBtn"),
    screenLabel: $("screenLabel"),
    switchBtn: $("switchBtn"),
    resetBtn: $("resetBtn"),
    fullBtn: $("fullBtn"),
    conf: $("conf"),
    confVal: $("confVal"),
    hint: $("hint"),
    totalCount: $("totalCount"),
    inView: $("inView"),
    latency: $("latency"),
    serverFps: $("serverFps"),
    resolution: $("resolution"),
    device: $("device"),
  };

  const overlayCtx = els.overlay.getContext("2d");
  const frameCanvas = document.createElement("canvas");
  const frameCtx = frameCanvas.getContext("2d", { alpha: false });

  const state = {
    running: false,
    starting: false,
    fatal: false,
    stream: null,
    ws: null,
    ready: false,
    inFlight: false,
    pendingSince: 0,
    capturing: false,
    timer: null,
    reconnectTimer: null,
    reconnectAttempt: 0,
    facing: "environment",
    mode: "camera",
    detections: [],
    counts: emptyCounts(),
    current: emptyCounts(),
    confidence: 0.35,
    backend: null,
    responseTimes: [],
    wakeLock: null,
    drawQueued: false,
  };

  function emptyCounts() {
    return { car: 0, motorcycle: 0, bus: 0, truck: 0, total: 0 };
  }

  function normalizeCounts(raw) {
    const out = emptyCounts();
    if (!raw || typeof raw !== "object") return out;
    for (const key of CLASS_KEYS) {
      const value = Number(raw[key]);
      out[key] = Number.isFinite(value) ? value : 0;
    }
    const total = Number(raw.total);
    out.total = Number.isFinite(total)
      ? total
      : CLASS_KEYS.reduce((sum, key) => sum + out[key], 0);
    return out;
  }

  function setStatus(kind, text) {
    els.statusPill.dataset.state = kind;
    els.statusText.textContent = text;
  }

  function setPlaceholder(title, text) {
    els.placeholderTitle.textContent = title;
    els.placeholderText.textContent = text;
  }

  function showAlert(message) {
    els.stageAlert.textContent = message;
    els.stageAlert.hidden = false;
  }

  function clearAlert() {
    els.stageAlert.hidden = true;
    els.stageAlert.textContent = "";
  }

  function applyBackend(backend) {
    if (!backend) return;
    state.backend = backend;
    const short = backend.short || backend.hardware || backend.label || "GPU";
    els.device.textContent = short;
    els.device.title = backend.label || short;
    els.backendBadge.textContent = backend.badge || short;
    els.backendBadge.title = backend.label || short;
  }

  function isLocalHost() {
    return ["localhost", "127.0.0.1", "::1", "[::1]"].includes(location.hostname);
  }

  function cameraContextOk() {
    return window.isSecureContext || isLocalHost();
  }

  // ------------------------------------------------------------------ camera

  function waitForEvent(target, event, timeout = 4000) {
    return new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        target.removeEventListener(event, finish);
        resolve();
      };
      target.addEventListener(event, finish, { once: true });
      setTimeout(finish, timeout);
    });
  }

  async function openCamera() {
    stopStream();

    const constraints = {
      audio: false,
      video: {
        facingMode: { ideal: state.facing },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      if (err && (err.name === "OverconstrainedError" || err.name === "TypeError")) {
        stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
      } else {
        throw err;
      }
    }

    state.stream = stream;
    els.video.srcObject = stream;
    await els.video.play().catch(() => {});
    if (!els.video.videoWidth) await waitForEvent(els.video, "loadedmetadata");
    resizeOverlay();
  }

  async function openScreen() {
    stopStream();

    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      throw new Error("Unsupported");
    }

    const constraints = {
      audio: false,
      video: {
        displaySurface: "monitor",
        width: { ideal: 1920 },
        height: { ideal: 1080 },
        frameRate: { ideal: 30 },
      },
    };

    let stream;
    try {
      stream = await navigator.mediaDevices.getDisplayMedia(constraints);
    } catch (err) {
      if (err && (err.name === "OverconstrainedError" || err.name === "TypeError")) {
        stream = await navigator.mediaDevices.getDisplayMedia({ audio: false, video: true });
      } else {
        throw err;
      }
    }

    state.stream = stream;
    els.video.srcObject = stream;
    await els.video.play().catch(() => {});
    if (!els.video.videoWidth) await waitForEvent(els.video, "loadedmetadata");
    resizeOverlay();

    for (const track of stream.getVideoTracks()) {
      track.onended = () => {
        if (state.running && state.mode === "screen") stopCamera();
      };
    }
  }

  function stopStream() {
    if (state.stream) {
      for (const track of state.stream.getTracks()) {
        try {
          track.stop();
        } catch {
          /* ignore */
        }
      }
      state.stream = null;
    }
    els.video.srcObject = null;
    els.stage.style.removeProperty("--video-aspect");
    document.body.classList.remove("camera-live");
    clearOverlay();
  }

  function cameraErrorMessage(err) {
    const name = err && err.name ? err.name : "";
    if (name === "NotAllowedError" || name === "SecurityError") {
      return "Camera permission was blocked. Allow camera access for this site in your browser settings, then try again.";
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return "No camera was found on this device.";
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      return "The camera is already in use by another app. Close it and try again.";
    }
    if (!cameraContextOk()) {
      return "Phone cameras require HTTPS. Open the https:// address shown in the terminal and accept the certificate.";
    }
    return (err && err.message) || "Could not start the camera.";
  }

  // -------------------------------------------------------------- websocket

  function wsOpen() {
    return !!state.ws && state.ws.readyState === WebSocket.OPEN;
  }

  function connectSocket() {
    if (state.ws && (state.ws.readyState === WebSocket.CONNECTING || state.ws.readyState === WebSocket.OPEN)) {
      return;
    }
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
    setStatus("connecting", "Connecting…");

    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    let ws;
    try {
      ws = new WebSocket(`${protocol}//${location.host}/ws/detect`);
    } catch {
      showAlert("Could not open the AI connection. Reload the page and retry.");
      return;
    }

    state.ws = ws;
    state.ready = false;
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      state.reconnectAttempt = 0;
    };

    ws.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      handleMessage(message);
    };

    ws.onclose = () => {
      const wasReady = state.ready;
      state.ready = false;
      state.inFlight = false;
      if (state.ws === ws) state.ws = null;

      if (state.running && !state.fatal) {
        setStatus("connecting", wasReady ? "Reconnecting…" : "Connecting…");
        scheduleReconnect();
      } else if (!state.running) {
        setStatus("idle", "Camera off");
      }
    };

    ws.onerror = () => {
      /* onclose handles recovery */
    };
  }

  function scheduleReconnect() {
    if (!state.running || state.fatal) return;
    clearTimeout(state.reconnectTimer);
    const delay = Math.min(5000, 700 * Math.pow(1.7, state.reconnectAttempt));
    state.reconnectAttempt += 1;
    state.reconnectTimer = setTimeout(connectSocket, delay);
  }

  function closeSocket() {
    clearTimeout(state.reconnectTimer);
    state.reconnectTimer = null;
    const ws = state.ws;
    state.ws = null;
    state.ready = false;
    if (ws) {
      ws.onclose = null;
      ws.onmessage = null;
      ws.onerror = null;
      try {
        ws.close(1000, "client stop");
      } catch {
        /* ignore */
      }
    }
  }

  function sendConfig() {
    if (wsOpen()) {
      state.ws.send(JSON.stringify({ type: "config", confidence: state.confidence }));
    }
  }

  // ------------------------------------------------------------ server events

  function handleMessage(message) {
    switch (message.type) {
      case "status": {
        const stateName = message.state;
        if (stateName === "loading") {
          setStatus("connecting", "Loading AI…");
          setPlaceholder("Loading AI model…", "Preparing the detector on the GPU. This can take a moment.");
        } else if (stateName === "warming") {
          setStatus("connecting", "Warming up…");
          setPlaceholder("Warming up the GPU…", message.message || "Getting the model ready.");
        }
        break;
      }
      case "ready": {
        state.ready = true;
        state.fatal = false;
        applyBackend(message.backend);
        document.body.classList.add("camera-live");
        setStatus("live", "Live");
        clearAlert();
        sendConfig();
        break;
      }
      case "result": {
        state.inFlight = false;
        applyResult(message);
        break;
      }
      case "reset": {
        state.counts = normalizeCounts(message.counts);
        state.current = normalizeCounts(message.current);
        state.detections = [];
        renderCounts();
        scheduleDraw();
        break;
      }
      case "config": {
        if (typeof message.confidence === "number") {
          state.confidence = message.confidence;
          els.conf.value = String(message.confidence);
          els.confVal.textContent = message.confidence.toFixed(2);
        }
        break;
      }
      case "error": {
        showAlert(message.message || "Something went wrong on the server.");
        if (message.fatal) {
          handleFatal(message.message || "The AI model is unavailable.");
        }
        break;
      }
      default:
        break;
    }
  }

  function handleFatal(message) {
    state.fatal = true;
    state.running = false;
    state.ready = false;
    stopPump();
    closeSocket();
    stopStream();
    releaseWakeLock();
    setStatus("error", "Unavailable");
    setPlaceholder("AI unavailable", message);
    updateButtons();
    els.fpsBadge.textContent = "0 FPS";
  }

  function applyResult(message) {
    const now = performance.now();
    state.responseTimes.push(now);
    while (state.responseTimes.length && now - state.responseTimes[0] > 2000) {
      state.responseTimes.shift();
    }

    state.inFlight = false;
    state.detections = Array.isArray(message.detections) ? message.detections : [];
    state.counts = normalizeCounts(message.counts);
    state.current = normalizeCounts(message.current);
    if (message.backend) applyBackend(message.backend);

    renderCounts();

    if (Number.isFinite(message.inference_ms)) {
      els.latency.textContent = `${Math.round(message.inference_ms)} ms`;
    }
    const fps = currentFps();
    els.serverFps.textContent = fps ? `${fps.toFixed(1)} fps` : "—";
    els.fpsBadge.textContent = `${fps ? Math.round(fps) : 0} FPS`;
    if (Array.isArray(message.frame) && message.frame.length === 2) {
      els.resolution.textContent = `${message.frame[0]}×${message.frame[1]}`;
    }

    scheduleDraw();
  }

  function currentFps() {
    const times = state.responseTimes;
    if (times.length < 2) return 0;
    const span = (times[times.length - 1] - times[0]) / 1000;
    if (span < 0.05) return 0;
    return (times.length - 1) / span;
  }

  // ---------------------------------------------------------------- counting

  let lastTotal = 0;

  function renderCounts() {
    const total = state.counts.total || 0;
    if (total !== lastTotal) {
      els.totalCount.textContent = String(total);
      if (total > lastTotal) {
        els.totalCount.classList.remove("bump");
        void els.totalCount.offsetWidth;
        els.totalCount.classList.add("bump");
      }
      lastTotal = total;
    }
    els.inView.textContent = String(state.current.total || 0);
    for (const key of CLASS_KEYS) {
      const el = document.querySelector(`[data-count="${key}"]`);
      if (el) el.textContent = String(state.counts[key] || 0);
    }
  }

  function requestReset() {
    state.counts = emptyCounts();
    state.current = emptyCounts();
    state.detections = [];
    lastTotal = 0;
    renderCounts();
    scheduleDraw();
    if (wsOpen()) state.ws.send(JSON.stringify({ type: "reset" }));
    els.resetBtn.classList.remove("is-spinning");
    void els.resetBtn.offsetWidth;
    els.resetBtn.classList.add("is-spinning");
  }

  // ------------------------------------------------------------------ frames

  function startPump() {
    stopPump();
    state.timer = setInterval(pump, SEND_INTERVAL_MS);
  }

  function stopPump() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    state.inFlight = false;
    state.capturing = false;
  }

  function pump() {
    if (!state.running || !state.ready || document.hidden) return;
    if (!wsOpen()) return;

    if (state.inFlight) {
      if (performance.now() - state.pendingSince > 6000) state.inFlight = false;
      return;
    }
    if (state.capturing) return;
    captureFrame();
  }

  function captureFrame() {
    const video = els.video;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh || video.readyState < 2) return;

    const scale = Math.min(1, MAX_SIDE / Math.max(vw, vh));
    const width = Math.max(2, Math.round(vw * scale));
    const height = Math.max(2, Math.round(vh * scale));

    if (frameCanvas.width !== width || frameCanvas.height !== height) {
      frameCanvas.width = width;
      frameCanvas.height = height;
    }

    try {
      frameCtx.drawImage(video, 0, 0, width, height);
    } catch {
      return;
    }

    state.capturing = true;
    frameCanvas.toBlob(
      (blob) => {
        state.capturing = false;
        if (!blob || !state.running) return;
        const ws = state.ws;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        state.inFlight = true;
        state.pendingSince = performance.now();
        try {
          ws.send(blob);
        } catch {
          state.inFlight = false;
        }
      },
      "image/jpeg",
      JPEG_QUALITY,
    );
  }

  // ------------------------------------------------------------------ canvas

  function resizeOverlay() {
    const rect = els.stage.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const vw = els.video.videoWidth;
    const vh = els.video.videoHeight;
    if (vw && vh && state.stream) {
      els.stage.style.setProperty("--video-aspect", `${vw} / ${vh}`);
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    els.overlay.width = Math.round(rect.width * dpr);
    els.overlay.height = Math.round(rect.height * dpr);
    scheduleDraw();
  }

  function videoRect() {
    const vw = els.video.videoWidth || 16;
    const vh = els.video.videoHeight || 9;
    const cw = els.overlay.width;
    const ch = els.overlay.height;
    const scale = Math.min(cw / vw, ch / vh);
    const width = vw * scale;
    const height = vh * scale;
    return { x: (cw - width) / 2, y: (ch - height) / 2, w: width, h: height };
  }

  function scheduleDraw() {
    if (state.drawQueued) return;
    state.drawQueued = true;
    requestAnimationFrame(() => {
      state.drawQueued = false;
      drawOverlay();
    });
  }

  function clearOverlay() {
    if (els.overlay.width) {
      overlayCtx.clearRect(0, 0, els.overlay.width, els.overlay.height);
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    const radius = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  }

  function drawOverlay() {
    const ctx = overlayCtx;
    const cw = els.overlay.width;
    const ch = els.overlay.height;
    if (!cw || !ch) return;
    ctx.clearRect(0, 0, cw, ch);
    if (!state.detections.length) return;

    const rect = videoRect();
    const cssWidth = els.overlay.getBoundingClientRect().width || 1;
    const dpr = cw / cssWidth;
    const lineWidth = Math.max(1.5, 2.1 * dpr);

    ctx.lineJoin = "round";
    ctx.textBaseline = "middle";
    ctx.font = `650 ${11 * dpr}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`;

    for (const det of state.detections) {
      if (!det || !Array.isArray(det.box) || det.box.length !== 4) continue;
      const box = det.box;
      let x1 = rect.x + Number(box[0]) * rect.w;
      let y1 = rect.y + Number(box[1]) * rect.h;
      let x2 = rect.x + Number(box[2]) * rect.w;
      let y2 = rect.y + Number(box[3]) * rect.h;
      if (![x1, y1, x2, y2].every(Number.isFinite)) continue;

      x1 = Math.max(rect.x, Math.min(x1, rect.x + rect.w));
      x2 = Math.max(rect.x, Math.min(x2, rect.x + rect.w));
      y1 = Math.max(rect.y, Math.min(y1, rect.y + rect.h));
      y2 = Math.max(rect.y, Math.min(y2, rect.y + rect.h));

      const w = Math.max(1, x2 - x1);
      const h = Math.max(1, y2 - y1);
      const color = CLASS_COLORS[det.cls] || "#22d3ee";

      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
      ctx.shadowBlur = 3 * dpr;
      roundRect(ctx, x1, y1, w, h, 7 * dpr);
      ctx.stroke();
      ctx.shadowBlur = 0;

      const label = `${String(det.cls || "").toUpperCase()}${det.id != null ? ` #${det.id}` : ""}`;
      const padX = 6 * dpr;
      const labelH = 17 * dpr;
      const textW = ctx.measureText(label).width;
      let lx = x1;
      let ly = y1 - labelH - 3 * dpr;
      if (ly < rect.y + 1) ly = y1 + 3 * dpr;
      if (lx + textW + padX * 2 > rect.x + rect.w) {
        lx = rect.x + rect.w - textW - padX * 2;
      }
      lx = Math.max(rect.x, lx);

      ctx.fillStyle = color;
      roundRect(ctx, lx, ly, textW + padX * 2, labelH, 4 * dpr);
      ctx.fill();

      ctx.fillStyle = "#04131a";
      ctx.fillText(label, lx + padX, ly + labelH / 2 + 0.5 * dpr);
    }
  }

  // ------------------------------------------------------------- start/stop

  function updateButtons() {
    const isCamera = state.mode === "camera";
    const isScreen = state.mode === "screen";

    els.startBtn.classList.toggle("is-stop", state.running && isCamera);
    els.startLabel.textContent = state.running && isCamera
      ? "Stop camera"
      : state.starting && isCamera
        ? "Starting…"
        : "Start camera";
    els.startBtn.disabled = state.starting || (state.running && !isCamera);

    els.screenBtn.classList.toggle("is-stop", state.running && isScreen);
    els.screenLabel.textContent = state.running && isScreen ? "Stop" : "Screen";
    els.screenBtn.disabled = state.starting || (state.running && !isScreen);

    els.switchBtn.style.display = isScreen ? "none" : "";
  }

  async function startCamera() {
    if (state.starting) return;
    state.starting = true;
    state.fatal = false;
    state.responseTimes = [];
    els.fpsBadge.textContent = "0 FPS";
    clearAlert();
    updateButtons();

    if (!cameraContextOk()) {
      state.starting = false;
      updateButtons();
      const text =
        "Phone cameras require a secure HTTPS connection. Open the https:// address printed in the terminal (accept the certificate warning).";
      showAlert(text);
      setPlaceholder("HTTPS required", text);
      setStatus("error", "HTTPS required");
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      state.starting = false;
      updateButtons();
      showAlert("This browser does not support camera access.");
      setPlaceholder("Browser unsupported", "Use Chrome, Safari, Edge, or Firefox.");
      setStatus("error", "Unsupported");
      return;
    }

    setStatus("connecting", "Starting camera…");
    setPlaceholder("Starting camera…", "Allow camera access when your browser asks.");

    try {
      state.mode = "camera";
      await openCamera();
      state.running = true;
      updateButtons();
      document.body.classList.add("camera-live");
      connectSocket();
      startPump();
      requestWakeLock();
      if (!state.ready) {
        setStatus("connecting", "Connecting to AI…");
        setPlaceholder("Connecting to the AI…", "The model is loading onto the GPU.");
      }
    } catch (err) {
      const message = cameraErrorMessage(err);
      showAlert(message);
      setPlaceholder("Camera unavailable", message);
      setStatus("error", "Camera error");
      state.running = false;
      stopStream();
      closeSocket();
      releaseWakeLock();
    } finally {
      state.starting = false;
      updateButtons();
    }
  }

  async function startScreen() {
    if (state.starting) return;
    state.starting = true;
    state.fatal = false;
    state.responseTimes = [];
    els.fpsBadge.textContent = "0 FPS";
    clearAlert();
    updateButtons();

    if (!cameraContextOk()) {
      state.starting = false;
      updateButtons();
      const text =
        "Screen capture requires a secure HTTPS connection. Open the https:// address printed in the terminal.";
      showAlert(text);
      setPlaceholder("HTTPS required", text);
      setStatus("error", "HTTPS required");
      return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
      state.starting = false;
      updateButtons();
      showAlert("This browser does not support screen capture. Use Chrome or Edge on a desktop.");
      setPlaceholder("Browser unsupported", "Use a desktop browser with screen capture support.");
      setStatus("error", "Unsupported");
      return;
    }

    setStatus("connecting", "Starting screen capture…");
    setPlaceholder("Starting screen capture…", "Choose Entire Screen in the browser prompt.");

    try {
      state.mode = "screen";
      await openScreen();
      state.running = true;
      updateButtons();
      document.body.classList.add("camera-live");
      connectSocket();
      startPump();
      requestWakeLock();
      if (!state.ready) {
        setStatus("connecting", "Connecting to the AI…");
        setPlaceholder("Connecting to the AI…", "The model is loading onto the compute device.");
      }
    } catch (err) {
      const message = err && err.message ? err.message : "Could not start screen capture.";
      showAlert(message === "Unsupported" ? "This browser does not support screen capture." : message);
      setPlaceholder("Capture unavailable", message);
      setStatus("error", "Capture error");
      state.running = false;
      state.mode = "camera";
      stopStream();
      closeSocket();
      releaseWakeLock();
    } finally {
      state.starting = false;
      updateButtons();
    }
  }

  function stopCamera() {
    state.running = false;
    state.ready = false;
    state.fatal = false;
    stopPump();
    closeSocket();
    stopStream();
    releaseWakeLock();
    state.detections = [];
    state.inFlight = false;
    state.responseTimes = [];
    els.fpsBadge.textContent = "0 FPS";
    els.serverFps.textContent = "—";
    setStatus("idle", "Camera off");
    setPlaceholder("Camera is off", "Tap Start camera and point your phone at the road.");
    clearAlert();
    updateButtons();
  }

  async function switchCamera() {
    const next = state.facing === "environment" ? "user" : "environment";
    if (!state.running) {
      state.facing = next;
      return;
    }
    const previous = state.facing;
    state.facing = next;
    try {
      await openCamera();
      document.body.classList.add("camera-live");
    } catch (err) {
      state.facing = previous;
      try {
        await openCamera();
      } catch {
        showAlert(cameraErrorMessage(err));
        stopCamera();
      }
    }
  }

  // ------------------------------------------------------------- wake lock

  async function requestWakeLock() {
    try {
      if ("wakeLock" in navigator && document.visibilityState === "visible") {
        state.wakeLock = await navigator.wakeLock.request("screen");
      }
    } catch {
      state.wakeLock = null;
    }
  }

  function releaseWakeLock() {
    try {
      if (state.wakeLock) state.wakeLock.release();
    } catch {
      /* ignore */
    }
    state.wakeLock = null;
  }

  // --------------------------------------------------------------- bootstrap

  async function bootstrap() {
    if (location.protocol !== "https:" && !isLocalHost()) {
      els.hint.textContent =
        "Phone cameras need HTTPS. Open the https:// link printed in the terminal and accept the certificate.";
      els.hint.classList.add("warn");
    }

    els.video.addEventListener("loadedmetadata", resizeOverlay);
    els.video.addEventListener("resize", resizeOverlay);

    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      const data = await response.json();
      if (data.backend) applyBackend(data.backend);
      if (data.error) {
        showAlert(`The AI model failed to load: ${data.error}`);
        setStatus("error", "Model error");
      }
    } catch {
      /* the WebSocket will report real errors */
    }

    updateButtons();
  }

  els.startBtn.addEventListener("click", () => {
    if (state.running) stopCamera();
    else startCamera();
  });
  els.screenBtn.addEventListener("click", () => {
    if (state.running && state.mode === "screen") stopCamera();
    else if (!state.running || state.mode !== "screen") startScreen();
  });
  els.switchBtn.addEventListener("click", switchCamera);
  els.resetBtn.addEventListener("click", requestReset);

  els.fullBtn.addEventListener("click", async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else if (els.viewer.requestFullscreen) {
        await els.viewer.requestFullscreen();
      }
    } catch {
      /* fullscreen is optional */
    }
  });

  if (!document.fullscreenEnabled && !document.webkitFullscreenEnabled) {
    els.fullBtn.style.display = "none";
  }

  els.conf.addEventListener("input", () => {
    state.confidence = Number(els.conf.value);
    els.confVal.textContent = state.confidence.toFixed(2);
  });
  els.conf.addEventListener("change", sendConfig);

  window.addEventListener("resize", resizeOverlay);
  window.addEventListener("orientationchange", () => setTimeout(resizeOverlay, 250));

  if ("ResizeObserver" in window) {
    new ResizeObserver(resizeOverlay).observe(els.stage);
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state.running) {
      state.inFlight = false;
      requestWakeLock();
      resizeOverlay();
    }
  });

  window.addEventListener("beforeunload", () => {
    closeSocket();
    stopStream();
    releaseWakeLock();
  });

  bootstrap();
})();
