# TrafficLens — live vehicle counter

A complete, self-hosted car counter: **Ultralytics YOLO11** on the backend, a mobile-first web
frontend, and real-time counting from any phone or laptop camera.

Open the page on your phone, press **Start camera**, and the phone streams frames over a
WebSocket. The server detects and tracks vehicles with ByteTrack, returns bounding boxes and
unique-ID counts, and the page draws the overlay live.

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
- Server-side GPU inference, client-side canvas overlay (low bandwidth)
- FPS, latency, input resolution, and active compute backend shown in the UI
- Adjustable confidence, reset counter, fullscreen, screen wake-lock
- Auto-reconnecting WebSocket and automatic HTTPS certificate for phone cameras

## GPU on this computer

This machine has **Intel UHD Graphics 770** and **no NVIDIA GPU**. Ultralytics cannot use CUDA
here, so the backend detects the Intel GPU and runs the model through **OpenVINO**
(`device="xpu"`). The first run exports `yolo11n.pt` to an OpenVINO model, then inference runs on
the Intel iGPU.

Device selection is automatic, in this order:

1. **CUDA** (`cuda:0`) if an NVIDIA GPU is ever available
2. **Intel GPU via OpenVINO** (`xpu`)
3. **CPU** fallback

The active backend is printed at startup, shown in the page header, and exposed at
`GET /api/status`. You can force one with `CAR_COUNTER_DEVICE` (see below).

## Quick start

The virtual environment and dependencies are installed in this project. From `car-counter/`:

```powershell
.\.venv\Scripts\python.exe run.py
```

or double-click **`start.bat`**.

The terminal prints the URLs:

```
This computer : https://localhost:8443
Your phone    : https://192.168.x.x:8443
```

Open the phone URL on a device connected to the **same network**, accept the self-signed
certificate warning once, then tap **Start camera**. The terminal also prints a QR code you can
scan.

> **Why HTTPS?** Mobile browsers only allow camera access on secure origins. The app generates a
> local self-signed certificate (`certs/`) automatically. Plain HTTP is still available with
> `--http` for localhost development, but phone cameras will not work over it.

### First-time phone certificate step

- **Android Chrome:** "Advanced" -> "Proceed to 192.168.x.x (unsafe)"
- **iOS Safari:** "Show Details" -> "visit this website" -> "Visit Website"
- Then press **Start camera** and allow camera permission.

Allow Python through the Windows Firewall on **private networks** when prompted, otherwise the
phone cannot reach the PC.

## Run in Vagrant (Ubuntu VM)

A `Vagrantfile` is included. It creates a VM named `trafficlens-vm`, syncs this project to
`/home/vagrant/car-counter`, installs CPU-only PyTorch + OpenVINO, and runs TrafficLens as a
systemd service.

```powershell
cd car-counter
vagrant up
```

URLs:

- **This PC:** https://localhost:8443 (forwarded port)
- **Host-only:** https://192.168.56.20:8443
- **Phone:** the VM's bridged DHCP address. Find it in the app log:

```powershell
vagrant ssh -c "journalctl -u trafficlens -n 40 --no-pager"
```

Look for the `Your phone` line (the bridged LAN address). The VM also has a bridged
adapter specifically so the phone can bypass the Windows firewall.

> **GPU note:** VirtualBox cannot pass the Intel UHD 770 into the guest. The Vagrant VM uses
> **CPU inference via OpenVINO**. For Intel GPU inference, stop the VM and run `run.py` on
> Windows instead (`vagrant halt`, then `.\.venv\Scripts\python.exe run.py`). Do not run both at
> once: they both want port 8443.

If corporate port security blocks the VM's MAC, remove the `public_network` line from the
`Vagrantfile` and use the forwarded port (with a Windows firewall rule) or a tunnel.

Useful VM commands:

```powershell
vagrant ssh
sudo systemctl status trafficlens
sudo systemctl restart trafficlens     # after editing code on the host
vagrant halt
vagrant destroy -f
```

The project folder is a synced folder, so edits made on Windows are immediately visible in the
VM. The app regenerates its HTTPS certificate on restart when the VM's IP addresses change.

## Install from scratch (another machine)

```powershell
cd car-counter
python -m venv .venv            # or: python -m virtualenv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

The first launch downloads `yolo11n.pt` (~5 MB) and exports it to OpenVINO. This can take a
minute. Weights and the exported model live in `models/`.

## Using the app

| Control | What it does |
| --- | --- |
| **Start camera** | Requests camera permission and opens the AI stream |
| **Switch camera** | Toggles rear/front camera (`facingMode`) |
| **Reset** | Starts a new counting session (does not stop the stream) |
| **Fullscreen** | Expands the video area |
| **Detection confidence** | Higher = fewer detections; lower = more sensitive |
| **Chip (top-left)** | Active backend, e.g. `GPU / OpenVINO` |
| **Chip (top-right)** | End-to-end processed FPS |

How counting works: every tracked vehicle ID is counted once per class. A vehicle that leaves
the frame and re-enters may be assigned a new ID and counted again; use **Reset** between
recording sessions.

## Configuration

Environment variables (PowerShell example: `$env:CAR_COUNTER_DEVICE="cpu"`):

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

```powershell
.\.venv\Scripts\python.exe run.py --port 9443
```

### Faster or more accurate?

- `yolo11n.pt` is the default balance of speed and accuracy for mobile/iGPU.
- For more accuracy on a strong GPU, export a larger model first:
  `.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11s.pt').export(format='openvino', imgsz=640)"`
  then set `CAR_COUNTER_MODEL=yolo11s.pt`.
- Lower `CAR_COUNTER_IMGSZ` (e.g. 480) for more FPS on weaker hardware.

## Verify the GPU backend

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py
```

It prints the selected device and average latency/FPS. On this PC the expected backend is
`Intel GPU / Intel(R) UHD Graphics 770 (iGPU) (OpenVINO)`, typically around 17-20 ms per frame.

## API

- `GET /api/status` — readiness, error, backend, model, active sessions
- `GET /api/health` — simple health check
- `WS /ws/detect` — binary JPEG frames in; JSON results out

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
├─ Vagrantfile       # Ubuntu VM with the app as a systemd service
├─ start.bat / start.ps1
├─ requirements.txt
└─ README.md
```

## Troubleshooting

**Phone cannot reach the server** — ensure phone and PC are on the same network, use the LAN IP
(not `localhost`), and allow Python on private networks in Windows Firewall. Guest/isolated
Wi-Fi networks often block device-to-device traffic.

**Camera permission blocked / no prompt** — the page must be HTTPS (or localhost). Accept the
certificate, reload, and allow camera access in site settings.

**`Address already in use`** — another server is on port 8443: `run.py --port 9443`.

**Model download/export fails** — check internet access and free disk space, then delete the
half-written file in `models/` and start again. Set `CAR_COUNTER_DEVICE=cpu` to skip OpenVINO.

**GPU not used** — run `scripts\benchmark.py`. If OpenVINO reports no GPU, update the Intel
graphics driver. You can force the backend with `CAR_COUNTER_DEVICE=xpu`; remove the exported
folder in `models/` to force a fresh OpenVINO export.

**Low FPS** — close other GPU-heavy apps, switch the phone to a lower camera resolution, lower
`CAR_COUNTER_IMGSZ`, or use a smaller model. The server processes one frame at a time per
camera, so latency never snowballs.

## Notes

- COCO vehicle classes are used (`car`, `motorcycle`, `bus`, `truck`). Bicycles/pedestrians are
  ignored.
- Detections on very small/distant vehicles can be missed; lower the confidence slider to
  increase sensitivity.
- This is a local-first tool: no cloud service, no video leaves your network.
