"""YOLO detector management and per-session tracking/counting."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config

logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    kind: str  # cuda | openvino | cpu
    label: str
    short: str
    badge: str
    device: str | None
    model_path: Path
    hardware: str

    def public(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "short": self.short,
            "badge": self.badge,
            "device": self.device,
            "hardware": self.hardware,
        }


def _friendly_error(exc: Exception) -> str:
    text = str(exc).strip() or exc.__class__.__name__
    return text.splitlines()[0][:400]


def _patch_openvino_cpu_config() -> None:
    """Use OpenVINO's throughput hint for CPU inference.

    Ultralytics hard-codes PERFORMANCE_HINT=LATENCY, which leaves a lot of CPU
    performance unused on virtual machines. This wrapper only changes CPU
    compilations; GPU inference is untouched.
    """
    try:
        import openvino as ov
    except Exception:  # noqa: BLE001 - optional backend
        return

    if getattr(ov.Core, "_trafficlens_cpu_patched", False):
        return

    original_compile = ov.Core.compile_model

    def compile_model(self, model, device_name=None, config=None, **kwargs):
        merged = dict(config or {})
        if str(device_name).upper() == "CPU":
            merged["PERFORMANCE_HINT"] = "THROUGHPUT"
        return original_compile(self, model, device_name, merged, **kwargs)

    ov.Core.compile_model = compile_model
    ov.Core._trafficlens_cpu_patched = True
    logger.info("OpenVINO CPU throughput mode enabled")


class DetectorManager:
    """Prepares the model once and hands out isolated detector sessions."""

    def __init__(self) -> None:
        self._init_lock = threading.Lock()
        self._started = False
        self._ready = threading.Event()
        self._sessions_lock = threading.Lock()
        self._sessions = 0
        self.error: str | None = None
        self.backend: BackendInfo | None = None

    # ------------------------------------------------------------------ setup

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set() and self.backend is not None and self.error is None

    def start(self) -> None:
        """Kick off model preparation in a background thread."""
        with self._init_lock:
            if self._started:
                return
            self._started = True
        thread = threading.Thread(target=self._run_init, name="trafficlens-model-init", daemon=True)
        thread.start()

    def wait_ready(self, timeout: float = 240.0) -> bool:
        return self._ready.wait(timeout)

    def _run_init(self) -> None:
        try:
            self.load()
        except Exception as exc:  # noqa: BLE001 - surface any startup failure to clients
            self.error = _friendly_error(exc)
            logger.exception("Model initialization failed")
        finally:
            self._ready.set()

    def load(self) -> None:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        weights = self._ensure_weights()
        override = config.DEVICE_OVERRIDE.lower()

        if override:
            logger.info("Device override requested: %s", override)
            if override.startswith("cuda"):
                self.backend = self._make_cuda_backend(weights, override)
            elif override.startswith(("intel", "xpu", "npu", "gpu")):
                self.backend = self._make_openvino_backend(weights, override)
            elif override == "cpu":
                self.backend = self._make_cpu_backend(weights)
            else:
                raise ValueError(
                    f"Unsupported CAR_COUNTER_DEVICE={override!r}. "
                    "Use cuda:0, xpu (Intel GPU), npu, or cpu."
                )
            if not self._validate_backend(self.backend):
                raise RuntimeError(
                    f"The requested device {override!r} could not run the model. "
                    "Remove CAR_COUNTER_DEVICE to let TrafficLens choose automatically."
                )
        else:
            self.backend = self._auto_backend(weights)

        logger.info("Using %s (%s) with %s", self.backend.label, self.backend.device, self.backend.model_path)
        self._ready.set()

    def _ensure_weights(self) -> Path:
        target = config.MODELS_DIR / config.MODEL_NAME
        if target.exists():
            return target

        logger.info("Downloading model %s ...", config.MODEL_NAME)
        from ultralytics import YOLO

        cwd = os.getcwd()
        os.chdir(config.MODELS_DIR)
        try:
            YOLO(config.MODEL_NAME)  # downloads the official checkpoint on first use
        finally:
            os.chdir(cwd)

        if not target.exists():
            candidates = list(config.MODELS_DIR.glob(Path(config.MODEL_NAME).stem + ".pt"))
            if candidates:
                return candidates[0]
            raise FileNotFoundError(f"Could not download model {config.MODEL_NAME}")
        return target

    def _auto_backend(self, weights: Path) -> BackendInfo:
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA is available")
                candidate = self._make_cuda_backend(weights, "cuda:0")
                if self._validate_backend(candidate):
                    return candidate
        except Exception as exc:  # noqa: BLE001 - optional backend
            logger.debug("CUDA probe failed: %s", exc)

        try:
            from openvino import Core

            devices = Core().available_devices
            gpu_devices = [d for d in devices if d.startswith("GPU")]
            if gpu_devices:
                logger.info("OpenVINO GPU devices detected: %s", gpu_devices)
                candidate = self._make_openvino_backend(weights, "xpu")
                if self._validate_backend(candidate):
                    return candidate
            else:
                logger.info("OpenVINO devices: %s (no Intel GPU found)", devices)
        except Exception as exc:  # noqa: BLE001 - optional backend
            logger.info("OpenVINO GPU unavailable: %s", _friendly_error(exc))

        return self._make_cpu_backend(weights)

    def _validate_backend(self, backend: BackendInfo) -> bool:
        """Run one real inference so broken drivers fail during startup, not mid-stream."""
        from ultralytics import YOLO

        if backend.kind == "openvino" and backend.device == "cpu":
            _patch_openvino_cpu_config()

        logger.info("Self-testing %s ...", backend.label)
        frame = np.full((360, 640, 3), 18, dtype=np.uint8)
        try:
            model = YOLO(str(backend.model_path), task="detect")
            kwargs = {
                "source": frame,
                "persist": True,
                "tracker": "bytetrack.yaml",
                "classes": config.VEHICLE_CLASS_IDS,
                "conf": 0.25,
                "imgsz": config.IMG_SIZE,
                "verbose": False,
            }
            if backend.device:
                kwargs["device"] = backend.device
            model.track(**kwargs)
            logger.info("Self-test passed for %s", backend.label)
            return True
        except Exception as exc:  # noqa: BLE001 - fall back to the next backend
            logger.warning("Self-test failed for %s: %s", backend.label, _friendly_error(exc))
            return False

    def _make_cuda_backend(self, weights: Path, device: str) -> BackendInfo:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA GPU is available")
        name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "NVIDIA GPU"
        return BackendInfo(
            kind="cuda",
            label=f"NVIDIA GPU · {name} (CUDA)",
            short="NVIDIA GPU",
            badge="GPU · CUDA",
            device=device,
            model_path=weights,
            hardware=name,
        )

    def _make_cpu_backend(self, weights: Path) -> BackendInfo:
        return BackendInfo(
            kind="cpu",
            label="CPU (no compatible GPU detected)",
            short="CPU",
            badge="CPU",
            device="cpu",
            model_path=weights,
            hardware="CPU",
        )

    def _make_openvino_backend(self, weights: Path, device: str) -> BackendInfo:
        from ultralytics import YOLO

        # Ultralytics maps PyTorch-style "xpu" / "npu" onto the Intel GPU / NPU for OpenVINO.
        normalized = device.strip().lower()
        if normalized in {"gpu", "gpu.0", "intel:gpu", "intel:gpu.0", "xpu", "xpu:0"}:
            normalized = "xpu"
        elif normalized in {"intel:cpu", "cpu"}:
            normalized = "cpu"
        elif normalized in {"intel:npu", "npu", "npu:0"}:
            normalized = "npu"
        is_gpu = normalized == "xpu"

        ov_dir = config.MODELS_DIR / f"{weights.stem}_openvino_model"
        if not any(ov_dir.glob("*.xml")):
            logger.info("Exporting %s to OpenVINO (first run only) ...", weights.name)
            base = YOLO(str(weights))
            exported = base.export(format="openvino", imgsz=config.IMG_SIZE, half=False)
            exported_path = Path(exported)
            if exported_path.is_dir():
                ov_dir = exported_path
            logger.info("OpenVINO model exported to %s", ov_dir)

        hardware = "Intel GPU"
        try:
            if is_gpu:
                from openvino import Core

                core = Core()
                gpu_ids = [d for d in core.available_devices if d.startswith("GPU")]
                if not gpu_ids:
                    raise RuntimeError("No Intel GPU is exposed by OpenVINO")
                try:
                    hardware = core.get_property(gpu_ids[0], "FULL_DEVICE_NAME") or hardware
                except Exception:  # noqa: BLE001 - cosmetic only
                    hardware = gpu_ids[0]
            elif normalized == "npu":
                hardware = "Intel NPU"
            else:
                hardware = "Intel CPU"
        except Exception as exc:  # noqa: BLE001 - fall back to CPU inference
            logger.warning("OpenVINO GPU init failed (%s); falling back to CPU", _friendly_error(exc))
            normalized = "cpu"
            is_gpu = False
            hardware = "Intel CPU"

        return BackendInfo(
            kind="openvino",
            label=f"Intel GPU · {hardware} (OpenVINO)" if is_gpu else f"{hardware} (OpenVINO)",
            short="Intel GPU" if is_gpu else "CPU",
            badge="GPU · OpenVINO" if is_gpu else "CPU · OpenVINO",
            device=normalized,
            model_path=ov_dir,
            hardware=hardware,
        )

    # --------------------------------------------------------------- sessions

    def acquire_session_slot(self) -> bool:
        with self._sessions_lock:
            if self._sessions >= config.MAX_SESSIONS:
                return False
            self._sessions += 1
            return True

    def release_session_slot(self) -> None:
        with self._sessions_lock:
            self._sessions = max(0, self._sessions - 1)

    def status(self) -> dict:
        return {
            "ready": self.is_ready,
            "loading": not self._ready.is_set(),
            "error": self.error,
            "backend": self.backend.public() if self.backend else None,
            "model": config.MODEL_NAME,
            "img_size": config.IMG_SIZE,
            "confidence": config.CONFIDENCE,
            "classes": config.CLASS_NAMES,
            "sessions": self._sessions,
            "max_sessions": config.MAX_SESSIONS,
        }


class DetectorSession:
    """Owns a YOLO instance plus tracker state for one connected camera."""

    def __init__(self, backend: BackendInfo) -> None:
        from ultralytics import YOLO

        if backend.kind == "openvino" and backend.device == "cpu":
            _patch_openvino_cpu_config()

        self.backend = backend
        self.confidence = config.CONFIDENCE
        self._lock = threading.Lock()
        self._counted: set[str] = set()
        self._counts = config.empty_counts()
        self._last_frame_at: float | None = None
        self._fps = 0.0
        self._closed = False

        logger.info("Loading %s for a new camera session", backend.model_path)
        self.model = YOLO(str(backend.model_path), task="detect")
        self._warmed_up = False

    # ------------------------------------------------------------------ warmup

    def warmup(self) -> None:
        """Run one blank frame so the first real frame is not slow."""
        blank = np.full((360, 640, 3), 24, dtype=np.uint8)
        try:
            self.process_frame(blank, count=False)
        finally:
            self._last_frame_at = None
            self._fps = 0.0
            self._warmed_up = True

    def close(self) -> None:
        self._closed = True
        try:
            self.model.predictor = None  # type: ignore[assignment]
        except Exception:  # noqa: BLE001 - best effort cleanup
            pass
        self._counted.clear()

    # ---------------------------------------------------------------- counting

    def set_confidence(self, value: float) -> float:
        self.confidence = min(max(float(value), 0.05), 0.95)
        return self.confidence

    def reset(self) -> None:
        with self._lock:
            self._counted.clear()
            self._counts = config.empty_counts()
            self._last_frame_at = None
            self._fps = 0.0
            # Clear tracker state as well so vehicles in view get fresh ids.
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) or []
            for tracker in trackers:
                reset_fn = getattr(tracker, "reset", None)
                if callable(reset_fn):
                    try:
                        reset_fn()
                    except Exception:  # noqa: BLE001 - optional API
                        pass

    # --------------------------------------------------------------- inference

    def process_jpeg(self, data: bytes) -> dict:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the camera frame")
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray, count: bool = True) -> dict:
        if self._closed:
            raise RuntimeError("Detector session is closed")

        with self._lock:
            height, width = frame.shape[:2]
            started = time.perf_counter()

            kwargs = {
                "source": frame,
                "persist": True,
                "tracker": "bytetrack.yaml",
                "classes": config.VEHICLE_CLASS_IDS,
                "conf": self.confidence,
                "iou": config.IOU,
                "imgsz": config.IMG_SIZE,
                "verbose": False,
            }
            if self.backend.device:
                kwargs["device"] = self.backend.device

            results = self.model.track(**kwargs)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            result = results[0] if results else None
            boxes = getattr(result, "boxes", None)
            detections: list[dict] = []
            current = config.empty_counts()

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                confidences = boxes.conf.detach().cpu().numpy()
                ids = None
                if getattr(boxes, "id", None) is not None:
                    ids = boxes.id.detach().cpu().numpy().astype(int)

                for i in range(len(xyxy)):
                    class_id = int(classes[i])
                    name = config.VEHICLE_CLASSES.get(class_id)
                    if name is None:
                        continue

                    track_id = int(ids[i]) if ids is not None and i < len(ids) else None
                    confidence = float(confidences[i])
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])

                    current[name] += 1
                    if count and track_id is not None:
                        key = f"{name}:{track_id}"
                        if key not in self._counted:
                            self._counted.add(key)
                            self._counts[name] += 1

                    detections.append(
                        {
                            "id": track_id,
                            "cls": name,
                            "conf": round(confidence, 3),
                            "box": [
                                round(x1 / width, 5),
                                round(y1 / height, 5),
                                round(x2 / width, 5),
                                round(y2 / height, 5),
                            ],
                        }
                    )

            now = time.perf_counter()
            if self._last_frame_at is not None:
                instant = 1.0 / max(now - self._last_frame_at, 1e-6)
                self._fps = instant if self._fps <= 0 else (self._fps * 0.8 + instant * 0.2)
            self._last_frame_at = now

            if count:
                self._counts["total"] = sum(self._counts[name] for name in config.CLASS_NAMES)
            current["total"] = sum(current[name] for name in config.CLASS_NAMES)

            return {
                "type": "result",
                "counts": dict(self._counts),
                "current": current,
                "detections": detections,
                "inference_ms": round(elapsed_ms, 1),
                "fps": round(self._fps, 1) if self._fps > 0 else 0.0,
                "frame": [width, height],
                "confidence": self.confidence,
            }
