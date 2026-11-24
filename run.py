#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson-Face-Tracker Launcher.
Starts FastAPI backend server with MJPEG live stream and Active Vision Servo Tracking.
Usage:
    sudo python3 run.py [--host 0.0.0.0] [--port 8000]
"""

import os
import sys
import argparse
import uvicorn

# Ensure repo root is in python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from web.app import app, init_system

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jetson Face Tracker Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding IP address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    args = parser.parse_args()

    print("=" * 68)
    print(" 🚀 JETSON FACE TRACKER - ACTIVE VISION GIMBAL SYSTEM")
    print("=" * 68)
    print(f"[*] Access Web UI at: http://<jetson-ip>:{args.port}/")
    print(f"[*] Default Mode: MANUAL (Use WASD / Arrow keys or On-Screen D-Pad)")
    print(f"[*] Safety Limits: Pan 0°~180°, Tilt 45°~135°")
    print("=" * 68)

    init_system()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", ws="none")
