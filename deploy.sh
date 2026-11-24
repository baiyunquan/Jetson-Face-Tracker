#!/usr/bin/env bash
# Deploy & setup script for Jetson Nano
set -e

echo "=== Setting up Jetson-Face-Tracker on Jetson Nano ==="

# 1. Install pip dependencies
pip3 install fastapi uvicorn pydantic

# 2. Update submodules
git submodule update --init --recursive

echo "=== Setup complete! ==="
echo "Run with: sudo python3 run.py --host 0.0.0.0 --port 8000"
