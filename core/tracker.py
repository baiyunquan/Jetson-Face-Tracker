#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Active Vision Tracking Engine for Jetson-Face-Tracker.
Features:
  - FOV-based geometric active vision with shutter timestamp phase compensation.
  - 3-second lost target timeout with smooth automatic recentering (90°, 90°).
  - Dual-mode operation: MANUAL (default) and AUTO (face lock).
  - High-performance HUD annotation for MJPEG streaming.
"""

import time
import threading
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from core.camera import CameraStream
from core.detector import UltraLightFaceDetector
from core.servo_jetson import ServoMover


class FaceTracker:
    """Orchestrates Camera, Detector, Servo Mover and Tracking State Machine."""

    def __init__(self,
                 camera: CameraStream,
                 detector: UltraLightFaceDetector,
                 servo: ServoMover,
                 fov_pan: float = 65.0,
                 fov_tilt: float = 50.0,
                 deadzone_px: int = 15,
                 lost_timeout_s: float = 3.0):
        self.camera = camera
        self.detector = detector
        self.servo = servo
        self.fov_pan = float(fov_pan)
        self.fov_tilt = float(fov_tilt)
        self.deadzone_px = deadzone_px
        self.lost_timeout_s = lost_timeout_s

        # Mode: 'manual' or 'auto' (Default is MANUAL)
        self.mode = "manual"
        self.manual_speed = 60.0  # dps

        # State tracking
        self.last_seen_time = time.monotonic()
        self.target_locked = False
        self.recentered = True
        self.fps = 0.0
        self.annotated_frame = None
        self.frame_lock = threading.Lock()

        # Tuning
        self.gain_pan = 0.85
        self.gain_tilt = 0.85

        self.running = True
        self.thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self.thread.start()

    def set_mode(self, mode: str):
        """Sets 'manual' or 'auto'."""
        if mode in ("manual", "auto"):
            self.mode = mode
            if mode == "manual":
                # Immediately halt auto tracking
                self.target_locked = False

    def manual_move(self, pan_dir: float, tilt_dir: float, speed_dps: float = None):
        """Called by manual UI to jog servos."""
        if self.mode != "manual":
            return
        speed = speed_dps if speed_dps is not None else self.manual_speed
        self.servo.jog(pan_dir, tilt_dir, speed)

    def center(self):
        """Smoothly centers gimbal to (90, 90)."""
        self.servo.center()
        self.recentered = True

    def _tracking_loop(self):
        last_time = time.monotonic()
        frame_count = 0
        fps_start = time.monotonic()

        while self.running:
            t_photo, frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            cx_screen, cy_screen = w // 2, h // 2

            # Run face detector
            faces = self.detector.detect(frame)
            primary_face = self.detector.select_primary_face(faces)

            now = time.monotonic()
            frame_count += 1
            if now - fps_start >= 1.0:
                self.fps = frame_count / (now - fps_start)
                frame_count = 0
                fps_start = now

            # Process state machine
            if primary_face is not None:
                self.last_seen_time = now
                self.target_locked = True
                self.recentered = False

                if self.mode == "auto":
                    # Error in pixels
                    fcx, fcy = primary_face['center']
                    ex = fcx - cx_screen
                    ey = fcy - cy_screen

                    # Deadband check
                    if abs(ex) > self.deadzone_px or abs(ey) > self.deadzone_px:
                        # Geometric angle offset
                        d_pan = -(ex / float(w)) * self.fov_pan * self.gain_pan
                        d_tilt = -(ey / float(h)) * self.fov_tilt * self.gain_tilt

                        # Historical timestamp alignment (compensate exposure latency)
                        hist_pan, hist_tilt = self.servo.position_at(t_photo)

                        new_target_pan = hist_pan + d_pan
                        new_target_tilt = hist_tilt + d_tilt

                        self.servo.set_target(new_target_pan, new_target_tilt)
            else:
                self.target_locked = False
                if self.mode == "auto":
                    # Check 3.0s lost timeout
                    if now - self.last_seen_time > self.lost_timeout_s:
                        if not self.recentered:
                            self.servo.center(speed_dps=50.0)
                            self.recentered = True

            # Render HUD overlay on frame
            if cv2 is not None:
                self._draw_hud(frame, faces, primary_face, cx_screen, cy_screen)

            with self.frame_lock:
                self.annotated_frame = frame

    def _draw_hud(self, frame, faces, primary_face, cx_screen, cy_screen):
        h, w = frame.shape[:2]

        # Draw center crosshair
        cv2.drawMarker(frame, (cx_screen, cy_screen), (0, 255, 0), cv2.MARKER_CROSS, 20, 1)

        # Draw all detected faces
        for f in faces:
            rx, ry, rw, rh = f['box']
            is_target = (primary_face is not None and f['center'] == primary_face['center'])
            color = (0, 255, 0) if is_target else (180, 180, 180)
            thickness = 2 if is_target else 1
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, thickness)
            if is_target:
                cv2.circle(frame, f['center'], 4, (0, 0, 255), -1)
                cv2.putText(frame, f"LOCKED {f['score']:.2f}", (rx, ry - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # Telemetry Bar
        pan_cur, tilt_cur, pan_tgt, tilt_tgt = self.servo.get_position()
        mode_text = f"MODE: {self.mode.upper()}"
        mode_color = (0, 255, 0) if self.mode == "auto" else (0, 200, 255)

        # Top Banner
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        cv2.putText(frame, mode_text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2, cv2.LINE_AA)

        status_txt = f"Pan: {pan_cur:.1f}deg (Tgt:{pan_tgt:.1f}) | Tilt: {tilt_cur:.1f}deg | FPS: {self.fps:.1f}"
        cv2.putText(frame, status_txt, (150, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    def get_latest_frame_jpeg(self):
        """Encodes annotated frame to JPEG for MJPEG streaming."""
        with self.frame_lock:
            if self.annotated_frame is None:
                return None
            if cv2 is not None:
                ret, jpeg = cv2.imencode('.jpg', self.annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ret:
                    return jpeg.tobytes()
        return None

    def get_status(self):
        pan_cur, tilt_cur, pan_tgt, tilt_tgt = self.servo.get_position()
        return {
            "mode": self.mode,
            "pan": round(pan_cur, 1),
            "tilt": round(tilt_cur, 1),
            "target_pan": round(pan_tgt, 1),
            "target_tilt": round(tilt_tgt, 1),
            "target_locked": self.target_locked,
            "fps": round(self.fps, 1),
            "manual_speed": self.manual_speed
        }

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
