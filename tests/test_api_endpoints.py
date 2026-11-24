#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import urllib.request
import json
import time

BASE_URL = "http://127.0.0.1:8080"


def post_json(endpoint: str, data: dict):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(endpoint: str):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("Testing Jetson-Face-Tracker API Endpoints...")

    # 1. Status
    status = get_json("/api/status")
    print("[1] Initial Status:", status)
    assert status["mode"] == "manual"
    assert status["pan"] == 90.0
    assert status["tilt"] == 90.0

    # 2. Mode switch to AUTO
    res_mode = post_json("/api/mode", {"mode": "auto"})
    print("[2] Switch to AUTO:", res_mode)
    assert res_mode["mode"] == "auto"

    status_auto = get_json("/api/status")
    assert status_auto["mode"] == "auto"

    # 3. Mode switch back to MANUAL
    res_manual = post_json("/api/mode", {"mode": "manual"})
    print("[3] Switch to MANUAL:", res_manual)
    assert res_manual["mode"] == "manual"

    # 4. Manual Jogging
    res_jog = post_json("/api/manual_move", {"pan": 1.0, "tilt": 0.0, "speed": 50.0})
    print("[4] Jogging Pan Left:", res_jog)
    time.sleep(0.3)
    post_json("/api/manual_move", {"pan": 0.0, "tilt": 0.0})

    status_after_jog = get_json("/api/status")
    print("[4] Status after jog:", status_after_jog)

    # 5. Center
    res_center = post_json("/api/center", {})
    print("[5] Center:", res_center)
    time.sleep(0.5)

    status_centered = get_json("/api/status")
    print("[5] Status centered:", status_centered)

    print("\n✅ ALL API ENDPOINTS VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
