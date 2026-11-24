#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.camera import CameraStream


def main():
    print("Testing Jetson Nano IMX708 Camera GStreamer Stream...")
    cam = CameraStream(sensor_id=0, capture_w=1536, capture_h=864, output_w=640, output_h=360, framerate=30)
    time.sleep(1.5)  # Let pipeline start

    t, frame = cam.read()
    if frame is not None and frame.shape == (360, 640, 3):
        print(f"[OK] Camera captured valid frame! Shape: {frame.shape}, Synthetic: {cam.is_synthetic}")
    else:
        print("[FAIL] Frame capture failed!")
        sys.exit(1)

    cam.stop()
    print("[OK] Camera stream stopped successfully.")


if __name__ == "__main__":
    main()
