#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.detector import UltraLightFaceDetector


def main():
    print("Testing Ultra-Light Face Detector on Jetson Nano...")
    det = UltraLightFaceDetector()
    if det.net is None:
        print("[FAIL] Network not loaded!")
        sys.exit(1)

    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    # Warmup
    det.detect(dummy)

    # 10 iterations
    t0 = time.time()
    n = 10
    for _ in range(n):
        det.detect(dummy)
    t1 = time.time()

    elapsed = t1 - t0
    latency_ms = (elapsed / n) * 1000.0
    fps = n / elapsed
    print(f"[OK] Avg latency: {latency_ms:.2f} ms per frame ({fps:.1f} FPS)")


if __name__ == "__main__":
    main()
