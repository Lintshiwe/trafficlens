"""FastAPI application: REST status plus a WebSocket inference channel."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from . import config
from .detector import DetectorManager, DetectorSession

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("trafficlens")

manager = DetectorManager()


@asynccontextmanager
async def lifespan(_: FastAPI):
    manager.start()
    logger.info("TrafficLens backend starting (model preparation runs in the background)")
    yield


app = FastAPI(title="TrafficLens", version="1.0.0", lifespan=lifespan)


@app.get("/api/status")
async def api_status() -> dict:
    status = manager.status()
    status["server_time"] = time.time()
    return status


@app.get("/api/health")
async def api_health() -> dict:
    return {"ok": manager.is_ready, "error": manager.error}


def _error_payload(message: str, fatal: bool = False) -> dict:
    return {"type": "error", "message": message, "fatal": fatal}


@app.websocket("/ws/detect")
async def detect_socket(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        if not manager.is_ready:
            await websocket.send_json({"type": "status", "state": "loading"})
            ready = await asyncio.to_thread(manager.wait_ready, 240.0)
            if not ready or manager.backend is None:
                detail = manager.error or "The AI model did not finish loading in time."
                await websocket.send_json(_error_payload(f"Model unavailable: {detail}", fatal=True))
                await websocket.close(code=1011)
                return

        if not manager.acquire_session_slot():
            await websocket.send_json(
                _error_payload(
                    f"Server is busy ({config.MAX_SESSIONS} camera sessions already running). "
                    "Close another tab and retry.",
                    fatal=True,
                )
            )
            await websocket.close(code=1013)
            return

        session: DetectorSession | None = None
        try:
            await websocket.send_json(
                {
                    "type": "status",
                    "state": "warming",
                    "message": f"Loading {config.MODEL_NAME} onto the {manager.backend.short}…",
                }
            )
            try:
                session = await asyncio.to_thread(DetectorSession, manager.backend)
                await asyncio.to_thread(session.warmup)
            except Exception as exc:  # noqa: BLE001 - report a useful failure to the browser
                logger.exception("Could not start a detector session")
                await websocket.send_json(
                    _error_payload(f"Could not start the detector: {exc}", fatal=True)
                )
                await websocket.close(code=1011)
                return

            await websocket.send_json(
                {
                    "type": "ready",
                    "backend": manager.backend.public(),
                    "model": config.MODEL_NAME,
                    "img_size": config.IMG_SIZE,
                    "confidence": session.confidence,
                    "classes": config.CLASS_NAMES,
                    "colors": config.CLASS_COLORS,
                }
            )
            logger.info("Camera session connected (%s)", manager.backend.short)

            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    break

                payload = message.get("bytes")
                if payload is not None:
                    if not payload:
                        continue
                    if len(payload) > config.MAX_FRAME_BYTES:
                        await websocket.send_json(
                            _error_payload(
                                f"Frame too large ({len(payload) / 1024 / 1024:.1f} MB). "
                                "Lower the camera resolution and retry."
                            )
                        )
                        continue
                    try:
                        result = await asyncio.to_thread(session.process_jpeg, payload)
                    except Exception as exc:  # noqa: BLE001 - keep the stream alive
                        logger.warning("Frame processing failed: %s", exc)
                        await websocket.send_json(_error_payload("That frame could not be processed."))
                        continue
                    await websocket.send_json(result)
                    continue

                text = message.get("text")
                if not text:
                    continue
                try:
                    command = json.loads(text)
                except json.JSONDecodeError:
                    continue

                kind = str(command.get("type", "")).lower()
                if kind == "reset":
                    session.reset()
                    await websocket.send_json(
                        {
                            "type": "reset",
                            "counts": config.empty_counts(),
                            "current": config.empty_counts(),
                        }
                    )
                elif kind == "config":
                    if "confidence" in command:
                        value = session.set_confidence(command["confidence"])
                        await websocket.send_json({"type": "config", "confidence": value})
                elif kind == "ping":
                    await websocket.send_json({"type": "pong", "t": time.time()})
        except WebSocketDisconnect:
            pass
        finally:
            if session is not None:
                await asyncio.to_thread(session.close)
            manager.release_session_slot()
            logger.info("Camera session closed")
    except WebSocketDisconnect:
        return
    except Exception:  # noqa: BLE001 - never crash the worker over one socket
        logger.exception("Unexpected WebSocket failure")
        with contextlib.suppress(Exception):
            await websocket.close(code=1011)


# Serve the mobile web client from the project's frontend folder.
config.FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(config.FRONTEND_DIR), html=True), name="frontend")
