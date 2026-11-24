#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Camera abstraction for Jetson Nano + IMX708 CSI camera.
Uses GStreamer nvarguscamerasrc with hardware ISP for real-time BGR acquisition.
Falls back to V4L2 or synthetic frames if camera is unavailable.
"""

import time
import threading
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class CameraStream:
    """Non-blocking background camera capture thread."""

    def __init__(self, sensor_id: int = 0,
                 capture_w: int = 1536, capture_h: int = 864,
                 output_w: int = 640, output_h: int = 360,
                 framerate: int = 30):
        self.output_w = output_w
        self.output_h = output_h
        self.framerate = framerate
        self.running = False
        self.cap = None
        self.frame = None
        self.timestamp = time.monotonic()
        self.lock = threading.Lock()
        self.is_synthetic = False

        if cv2 is not None:
            # GStreamer pipeline for Jetson Nano Tegra ISP + IMX708
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM), width={capture_w}, height={capture_h}, format=NV12, framerate={framerate}/1 ! "
                f"nvvidconv ! "
                f"video/x-raw, width={output_w}, height={output_h}, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
            )
            try:
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if not self.cap.isOpened():
                    # Fallback to standard V4L2 /dev/video0
                    self.cap = cv2.VideoCapture(0)
            except Exception:
                self.cap = None

        if self.cap is None or not self.cap.isOpened():
            self.is_synthetic = True
            print("[Camera] No physical camera found, running in Synthetic Mode.")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _generate_synthetic_frame(self, t: float) -> np.ndarray:
        """Generates a test pattern with a bouncing simulated face target."""
        frame = np.full((self.output_h, self.output_w, 3), 30, dtype=np.uint8)
        # Grid lines
        for x in range(0, self.output_w, 40):
            cv2.line(frame, (x, 0), (x, self.output_h), (45, 45, 45), 1)
        for y in range(0, self.output_h, 40):
            cv2.line(frame, (0, y), (self.output_w, y), (45, 45, 45), 1)

        # Center crosshair
        cx, cy = self.output_w // 2, self.output_h // 2
        cv2.line(frame, (cx - 15, cy), (cx + 15, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 15), (cx, cy + 15), (0, 255, 0), 1)

        # Simulated moving face box (oval)
        fx = int(cx + (self.output_w // 4) * np.sin(t * 1.5))
        fy = int(cy + (self.output_h // 5) * np.cos(t * 1.0))
        cv2.ellipse(frame, (fx, fy), (35, 45), 0, 0, 360, (200, 200, 200), -1)
        cv2.circle(frame, (fx - 12, fy - 10), 4, (40, 40, 40), -1)
        cv2.circle(frame, (fx + 12, fy - 10), 4, (40, 40, 40), -1)
        cv2.ellipse(frame, (fx, fy + 15), (15, 6), 0, 0, 180, (40, 40, 40), 2)

        cv2.putText(frame, "SYNTHETIC TARGET MODE", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _capture_loop(self):
        while self.running:
            now = time.monotonic()
            if not self.is_synthetic and self.cap is not None:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                        self.timestamp = now
                else:
                    time.sleep(0.01)
            else:
                if cv2 is not None:
                    frame = self._generate_synthetic_frame(now)
                    with self.lock:
                        self.frame = frame
                        self.timestamp = now
                time.sleep(1.0 / self.framerate)

    def read(self):
        """Returns (timestamp, frame)."""
        with self.lock:
            if self.frame is not None:
                return self.timestamp, self.frame.copy()
            else:
                # Blank initial frame
                dummy = np.zeros((self.output_h, self.output_w, 3), dtype=np.uint8)
                return self.timestamp, dummy

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()
