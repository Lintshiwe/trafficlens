"""Benchmark the configured compute backend without a camera.

Usage:
    .venv\\Scripts\\python.exe scripts\\benchmark.py [path\\to\\image.jpg]

The script prints the selected device and average inference time / FPS.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.detector import DetectorManager, DetectorSession  # noqa: E402


def main() -> int:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if image_path is not None and not image_path.exists():
        print(f"Image not found: {image_path}")
        return 1

    manager = DetectorManager()
    manager.load()
    backend = manager.backend
    assert backend is not None
    print(f"Backend : {backend.label}")
    print(f"Device  : {backend.device}")
    print(f"Model   : {backend.model_path}")

    session = DetectorSession(backend)
    session.warmup()

    if image_path is not None:
        import cv2

        frame = cv2.imread(str(image_path))
        if frame is None:
            print("Could not decode the image.")
            return 1
    else:
        frame = np.full((720, 1280, 3), 70, dtype=np.uint8)

    runs = 20
    started = time.perf_counter()
    for _ in range(runs):
        result = session.process_frame(frame, count=False)
    elapsed = (time.perf_counter() - started) / runs * 1000.0

    print("")
    print(f"Average : {elapsed:.1f} ms per frame")
    print(f"Speed   : {1000.0 / elapsed:.1f} FPS (synthetic stream)")
    print(f"Current : {result['current']['total']} vehicles visible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
