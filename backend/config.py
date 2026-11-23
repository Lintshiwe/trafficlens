"""Central configuration for TrafficLens."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("CAR_COUNTER_MODELS_DIR", str(ROOT_DIR / "models"))).resolve()
FRONTEND_DIR = ROOT_DIR / "frontend"
CERT_DIR = ROOT_DIR / "certs"

MODEL_NAME = os.getenv("CAR_COUNTER_MODEL", "yolo11n.pt")
IMG_SIZE = int(os.getenv("CAR_COUNTER_IMGSZ", "640"))

try:
    CONFIDENCE = min(max(float(os.getenv("CAR_COUNTER_CONF", "0.35")), 0.05), 0.95)
except ValueError:
    CONFIDENCE = 0.35

try:
    IOU = min(max(float(os.getenv("CAR_COUNTER_IOU", "0.5")), 0.05), 0.95)
except ValueError:
    IOU = 0.5

MAX_SESSIONS = max(1, int(os.getenv("CAR_COUNTER_MAX_SESSIONS", "4")))
DEVICE_OVERRIDE = os.getenv("CAR_COUNTER_DEVICE", "").strip()
MAX_FRAME_BYTES = max(1, int(os.getenv("CAR_COUNTER_MAX_FRAME_MB", "8"))) * 1024 * 1024
HOST = os.getenv("CAR_COUNTER_HOST", "0.0.0.0")
PORT = int(os.getenv("CAR_COUNTER_PORT", "8443"))

# COCO class ids for road vehicles.
VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
VEHICLE_CLASS_IDS = sorted(VEHICLE_CLASSES)
CLASS_NAMES = list(VEHICLE_CLASSES.values())

CLASS_COLORS: dict[str, str] = {
    "car": "#22d3ee",
    "motorcycle": "#f472b6",
    "bus": "#a78bfa",
    "truck": "#fb923c",
}


def empty_counts() -> dict[str, int]:
    """Return a fresh zeroed counts dictionary."""
    counts = {name: 0 for name in CLASS_NAMES}
    counts["total"] = 0
    return counts
