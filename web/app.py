#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastAPI Backend Server for Jetson-Face-Tracker.
Provides:
  - MJPEG low-latency streaming endpoint (/video_feed)
  - REST APIs for Mode Toggle, Manual Jogging, Speed Tuning, and Status.
  - Serves self-contained single-page Cyber HUD web interface.
"""

import os
import sys
import time
import asyncio
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add parent path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.camera import CameraStream
from core.detector import UltraLightFaceDetector
from core.servo_jetson import ServoMover
from core.tracker import FaceTracker

app = FastAPI(title="Jetson-Face-Tracker API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global hardware & tracking singletons
camera: Optional[CameraStream] = None
detector: Optional[UltraLightFaceDetector] = None
servo: Optional[ServoMover] = None
tracker: Optional[FaceTracker] = None


def init_system():
    global camera, detector, servo, tracker
    if tracker is not None:
        return

    print("[System] Initializing camera...")
    camera = CameraStream(sensor_id=0, capture_w=1536, capture_h=864, output_w=640, output_h=360, framerate=30)

    print("[System] Initializing face detector...")
    detector = UltraLightFaceDetector()

    print("[System] Initializing servo mover (Pan 0-180, Tilt 45-135)...")
    servo = ServoMover(pan_min=0.0, pan_max=180.0, pan_start=90.0,
                       tilt_min=45.0, tilt_max=135.0, tilt_start=90.0)

    print("[System] Starting Active Vision Tracker (Default: MANUAL)...")
    tracker = FaceTracker(camera=camera, detector=detector, servo=servo,
                          fov_pan=65.0, fov_tilt=50.0, deadzone_px=15, lost_timeout_s=3.0)
    tracker.set_mode("manual")


class ModeRequest(BaseModel):
    mode: str  # "manual" or "auto"


class MoveRequest(BaseModel):
    pan: float = 0.0    # -1 (left), 0 (stop), 1 (right)
    tilt: float = 0.0   # -1 (down), 0 (stop), 1 (up)
    speed: Optional[float] = None


class SpeedRequest(BaseModel):
    speed: float


@app.on_event("startup")
def startup_event():
    init_system()


@app.on_event("shutdown")
def shutdown_event():
    global tracker, servo, camera
    if tracker:
        tracker.stop()
    if servo:
        servo.stop()
    if camera:
        camera.stop()


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_file = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Jetson-Face-Tracker UI</h1><p>index.html missing.</p>"


@app.get("/video_feed")
async def video_feed():
    """MJPEG streaming endpoint with boundary multipart/x-mixed-replace."""
    async def frame_generator():
        last_id = None
        while True:
            if tracker is not None:
                jpeg_bytes = tracker.get_latest_frame_jpeg()
                if jpeg_bytes is not None and id(jpeg_bytes) != last_id:
                    last_id = id(jpeg_bytes)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
            await asyncio.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    if tracker:
        tracker.set_mode(req.mode.lower())
        return {"status": "ok", "mode": tracker.mode}
    return {"status": "error", "message": "tracker not ready"}


@app.post("/api/manual_move")
def manual_move(req: MoveRequest):
    if tracker:
        tracker.manual_move(pan_dir=req.pan, tilt_dir=req.tilt, speed_dps=req.speed)
        return {"status": "ok"}
    return {"status": "error"}


@app.post("/api/center")
def center_gimbal():
    if tracker:
        tracker.center()
        return {"status": "ok"}
    return {"status": "error"}


@app.post("/api/speed")
def set_speed(req: SpeedRequest):
    if tracker:
        tracker.manual_speed = max(10.0, min(150.0, req.speed))
        return {"status": "ok", "manual_speed": tracker.manual_speed}
    return {"status": "error"}


@app.get("/api/status")
def get_status():
    if tracker:
        return tracker.get_status()
    return {"status": "not_ready"}


if __name__ == "__main__":
    init_system()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info", ws="none")
