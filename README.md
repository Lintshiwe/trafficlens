# TrafficLens - live vehicle counter

A self-hosted car counter: **Ultralytics YOLO11** on the backend, a mobile-first web
frontend, and real-time counting from any phone or laptop camera.

Open the page on your phone, press **Start camera**, and the phone streams frames over a
secure WebSocket. The server detects and tracks vehicles with ByteTrack, returns bounding
boxes and unique-ID counts, and the page draws the overlay live.

```
Phone / laptop camera
        |  JPEG frames over WebSocket (wss)
        v
FastAPI --> YOLO11 + ByteTrack --> counts + detections
        ^
        +-- JSON results back to the browser overlay
```

## Features

- Live camera on **phone or desktop** (front/rear camera switch)
- YOLO11 vehicle detection: cars, trucks, buses, motorcycles
- ByteTrack unique-vehicle counting per session plus "in view now"
- Server-side inference, client-side canvas overlay (low bandwidth)
- FPS, latency, input resolution, and active compute backend shown in the UI
- Adjustable confidence, reset counter, fullscreen, screen wake-lock
- Screen capture mode: analyze whatever is shown on the monitor (no camera needed)
- Auto-reconnecting WebSocket and automatic HTTPS certificate for phone cameras

## GPU support

Device selection is automatic:

1. **CUDA** (`cuda:0`) if an NVIDIA GPU is available
2. **Intel GPU via OpenVINO** (`xpu`) on Intel hardware
3. **CPU** fallback

The active backend is printed at startup, shown in the page header, and exposed at
`GET /api/status`. You can force one with `CAR_COUNTER_DEVICE` (see Configuration).

## Requirements

- Python 3.10 or newer
- A modern browser with camera access (Chrome, Safari, Edge, Firefox)
- For phone use: the phone and the PC on the same network, or a tunnel

## Quick start

```bash
git clone https://github.com/Lintshiwe/trafficlens.git
cd trafficlens
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

Linux / macOS:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

The terminal prints the URLs:

```
This computer : https://localhost:8443
Your phone    : https://<your-lan-ip>:8443
```

Open the phone URL on a device connected to the **same network**, accept the self-signed
certificate warning once, then tap **Start camera**. The terminal also prints a QR code you
can scan.

> **Why HTTPS?** Mobile browsers only allow camera access on secure origins. The app
> generates a local self-signed certificate (`certs/`) automatically. Plain HTTP is still
> available with `--http` for localhost development, but phone cameras will not work over
> it.

### First-time phone certificate step

- **Android Chrome:** "Advanced" -> "Proceed to ... (unsafe)"
- **iOS Safari:** "Show Details" -> "visit this website" -> "Visit Website"
- Then press **Start camera** and allow camera permission.

Allow Python through your firewall on private networks when prompted, otherwise the phone
cannot reach the PC.

## Install notes

The first launch downloads `yolo11n.pt` (~5 MB) and exports it to OpenVINO. This can take
a minute on the first run. Weights and the exported model live in `models/`.

## Using the app

Screen mode runs on the machine whose screen you want to capture. Use the Windows host
(not the Vagrant VM) so the GPU is available, and keep the app page open while it runs.

| Control | What it does |
| --- | --- |
| **Start camera** | Requests camera permission and opens the AI stream |
| **Switch camera** | Toggles rear/front camera (`facingMode`) |
| **Screen** | Captures the whole monitor (pick "Entire Screen" in the browser prompt) |
| **Reset** | Starts a new counting session (does not stop the stream) |
| **Fullscreen** | Expands the video area |
| **Detection confidence** | Higher = fewer detections; lower = more sensitive |
| **Chip (top-left)** | Active backend, e.g. `GPU / OpenVINO` |
| **Chip (top-right)** | End-to-end processed FPS |

How counting works: every tracked vehicle ID is counted once per class. A vehicle that leaves
the frame and re-enters may be assigned a new ID and counted again; use **Reset** between
recording sessions.

## Configuration

Environment variables (example: `CAR_COUNTER_DEVICE=cpu`):

| Variable | Default | Description |
| --- | --- | --- |
| `CAR_COUNTER_MODEL` | `yolo11n.pt` | Ultralytics checkpoint name/path |
| `CAR_COUNTER_IMGSZ` | `640` | Inference input size |
| `CAR_COUNTER_CONF` | `0.35` | Default confidence threshold |
| `CAR_COUNTER_IOU` | `0.5` | NMS IoU threshold |
| `CAR_COUNTER_DEVICE` | auto | `cuda:0`, `xpu`, `npu`, or `cpu` |
| `CAR_COUNTER_MAX_SESSIONS` | `4` | Concurrent camera sessions |
| `CAR_COUNTER_MAX_FRAME_MB` | `8` | Max accepted JPEG frame size |
| `CAR_COUNTER_PORT` | `8443` | Default HTTPS port |
| `CAR_COUNTER_MODELS_DIR` | `./models` | Model storage directory |

Run with a different port:

```bash
python run.py --port 9443
```

### Faster or more accurate?

- `yolo11n.pt` is the default balance of speed and accuracy.
- For more accuracy on a strong GPU, export a larger model first:
  `python -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='openvino', imgsz=640)"`
  then set `CAR_COUNTER_MODEL=yolo11s.pt`.
- Lower `CAR_COUNTER_IMGSZ` (e.g. 480) for more FPS on weaker hardware.

## Verify the backend

```bash
python scripts/benchmark.py
```

It prints the selected device and average latency/FPS. The expected backend depends on your
hardware (CUDA, Intel GPU via OpenVINO, or CPU).

## API

- `GET /api/status` - readiness, error, backend, model, active sessions
- `GET /api/health` - simple health check
- `WS /ws/detect` - binary JPEG frames in; JSON results out

WebSocket protocol:

```jsonc
// server -> client on connect
{ "type": "status", "state": "loading" | "warming" }
{ "type": "ready", "backend": {...}, "classes": ["car", ...], "colors": {...} }

// client -> server
//   binary: JPEG frame
{ "type": "reset" }
{ "type": "config", "confidence": 0.4 }

// server -> client after each frame
{
  "type": "result",
  "counts":  { "car": 12, "truck": 3, "bus": 0, "motorcycle": 1, "total": 16 },
  "current": { "car": 4, "truck": 1, "bus": 0, "motorcycle": 0, "total": 5 },
  "detections": [
    { "id": 7, "cls": "car", "conf": 0.91, "box": [0.12, 0.34, 0.48, 0.72] }
  ],
  "inference_ms": 38.4,
  "fps": 11.2,
  "frame": [640, 360],
  "confidence": 0.35
}
```

Bounding boxes are normalized `[x1, y1, x2, y2]` relative to the submitted frame.

## Project layout

```
car-counter/
├─ backend/
│  ├─ config.py      # env settings, class filters, paths
│  ├─ detector.py    # backend selection, OpenVINO export, tracking/counting
│  ├─ certs.py       # self-signed HTTPS certificate generation
│  └─ main.py        # FastAPI app, REST status, WebSocket session
├─ frontend/
│  ├─ index.html     # mobile-first UI
│  ├─ styles.css     # responsive dark theme
│  ├─ app.js         # camera capture, WebSocket, overlay drawing
│  ├─ favicon.svg
│  └─ manifest.webmanifest
├─ models/           # downloaded checkpoints + OpenVINO export
├─ scripts/
│  └─ benchmark.py   # backend/latency check
├─ certs/            # generated cert (created automatically)
├─ run.py            # HTTPS server launcher with QR code
├─ Vagrantfile       # optional Ubuntu VM with the app as a service
├─ start.bat / start.ps1
├─ requirements.txt
└─ README.md
```

## Vagrant (optional)

A `Vagrantfile` is included for running the app in an Ubuntu VM. The VM uses CPU inference
because VirtualBox cannot expose the host GPU. See the file for network and provisioning
details.

## Troubleshooting

**Phone cannot reach the server** - ensure phone and PC are on the same network, use the LAN IP
(not `localhost`), and allow Python on private networks in your firewall. Guest or isolated
Wi-Fi networks often block device-to-device traffic.

**Camera permission blocked / no prompt** - the page must be HTTPS (or localhost). Accept the
certificate, reload, and allow camera access in site settings.

**`Address already in use`** - another server is on port 8443: `run.py --port 9443`.

**Model download/export fails** - check internet access and free disk space, then delete the
half-written file in `models/` and start again. Set `CAR_COUNTER_DEVICE=cpu` to skip
OpenVINO.

**GPU not used** - run `scripts/benchmark.py`. If OpenVINO reports no GPU, update the Intel
graphics driver. You can force the backend with `CAR_COUNTER_DEVICE=xpu`; remove the exported
folder in `models/` to force a fresh OpenVINO export.

**Low FPS** - close other GPU-heavy apps, switch the phone to a lower camera resolution, lower
`CAR_COUNTER_IMGSZ`, or use a smaller model. The server processes one frame at a time per
camera, so latency never snowballs.

## Notes

- COCO vehicle classes are used (`car`, `motorcycle`, `bus`, `truck`). Bicycles and
  pedestrians are ignored.
- Detections on very small or distant vehicles can be missed; lower the confidence slider to
  increase sensitivity.
- This is a local-first tool: no cloud service, no video leaves your network.
