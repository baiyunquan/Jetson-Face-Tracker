#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Jetson-Face-Tracker Servo & Motion Controller
Hardware PWM backend for Jetson Nano (Pin 32 PWM0 & Pin 33 PWM2) with:
  - Automatic Tegra hardware register unlock (/dev/mem).
  - 100Hz background trapezoidal S-curve acceleration & braking profiler.
  - Strict hardware limits: Pan [0°, 180°], Tilt [45°, 135°].
  - Mock fallback for off-device testing.
"""

import os
import sys
import time
import math
import mmap
import struct
import threading
from collections import deque


def fix_jetson_hardware_registers():
    """Unlocks Jetson Nano Pin 32 (PWM0) and Pin 33 (PWM2) hardware registers."""
    if not os.path.exists('/dev/mem'):
        return
    try:
        with open('/dev/mem', 'r+b') as f:
            # 1. Base PINMUX = 0x70003000
            mem_pmx = mmap.mmap(f.fileno(), 4096, offset=0x70003000)
            mem_pmx[0x1fc:0x200] = struct.pack('<I', 0x45)  # Pin 32 -> PWM0
            mem_pmx[0x248:0x24c] = struct.pack('<I', 0x46)  # Pin 33 -> PWM2

            # 2. Base GPIO = 0x6000d000
            mem_gpio = mmap.mmap(f.fileno(), 4096, offset=0x6000d000)
            cnf_e = struct.unpack('<I', mem_gpio[0x100:0x104])[0]
            mem_gpio[0x100:0x104] = struct.pack('<I', cnf_e & ~0x40)  # Clear PE6 bit 6

            cnf_v = struct.unpack('<I', mem_gpio[0x504:0x508])[0]
            mem_gpio[0x504:0x508] = struct.pack('<I', cnf_v & ~0x01)  # Clear PV0 bit 0
    except Exception as e:
        # If running as non-root or non-Tegra, log warning
        pass


class SysfsPWMChannel:
    """Manages an individual sysfs PWM channel."""

    def __init__(self, chip_path: str, channel: int, pulse_min_us=500, pulse_max_us=2500):
        self.chip_path = chip_path
        self.channel = channel
        self.pulse_min_us = pulse_min_us
        self.pulse_max_us = pulse_max_us
        self.ch_dir = os.path.join(chip_path, f"pwm{channel}")
        self.enabled = False
        self.mock = not os.path.exists(chip_path)

        if not self.mock:
            if not os.path.exists(self.ch_dir):
                try:
                    with open(os.path.join(chip_path, "export"), "w") as f:
                        f.write(str(channel))
                except Exception:
                    pass
                time.sleep(0.05)

            # 50 Hz = 20,000,000 ns
            try:
                self._write("enable", 0)
                self._write("period", 20000000)
            except Exception:
                pass

    def _write(self, filename: str, val):
        if self.mock:
            return
        path = os.path.join(self.ch_dir, filename)
        try:
            with open(path, "w") as f:
                f.write(str(val))
        except Exception:
            pass

    def set_angle(self, deg: float):
        """Converts angle [0..180] to pulse width [500..2500us] and writes duty cycle."""
        deg = max(0.0, min(180.0, float(deg)))
        us = self.pulse_min_us + (self.pulse_max_us - self.pulse_min_us) * (deg / 180.0)
        duty_ns = int(round(us * 1000.0))
        if not self.mock:
            self._write("duty_cycle", duty_ns)
            if not self.enabled:
                self._write("enable", 1)
                self.enabled = True

    def release(self):
        """Disables PWM to release holding torque."""
        if not self.mock and self.enabled:
            self._write("enable", 0)
            self.enabled = False


class _AxisMotion:
    """Calculates smooth trapezoidal acceleration, cruising, and braking for one axis."""

    def __init__(self, name: str, start_deg: float, min_deg: float, max_deg: float,
                 speed_dps=180.0, accel_dps2=800.0):
        self.name = name
        self.min_deg = float(min_deg)
        self.max_deg = float(max_deg)
        self.pos = float(max(self.min_deg, min(self.max_deg, start_deg)))
        self.target = self.pos
        self.vel = 0.0
        self.speed = float(speed_dps)
        self.accel = float(accel_dps2)
        self.custom_speed = None

    def set_target(self, target_deg: float, custom_speed: float = None):
        clamped = max(self.min_deg, min(self.max_deg, float(target_deg)))
        self.target = clamped
        self.custom_speed = custom_speed

    def step(self, dt: float) -> float:
        dist = self.target - self.pos
        if abs(dist) < 0.05 and abs(self.vel) < 2.0:
            self.pos = self.target
            self.vel = 0.0
        else:
            top_speed = self.custom_speed if self.custom_speed is not None else self.speed
            # Target braking speed v = sqrt(2 * a * d)
            braking_speed = math.sqrt(2.0 * self.accel * abs(dist))
            want_vel = math.copysign(min(top_speed, braking_speed), dist)
            dv = max(-self.accel * dt, min(self.accel * dt, want_vel - self.vel))
            self.vel += dv
            new_pos = self.pos + self.vel * dt
            # If overshoot, clamp to target
            if (self.target - new_pos) * dist < 0:
                new_pos = self.target
                self.vel = 0.0
            self.pos = max(self.min_deg, min(self.max_deg, new_pos))
        return self.pos


class ServoMover:
    """
    100Hz dual-axis servo motion coordinator for Jetson Nano.
    Pan (Pin 32): 0° ~ 180° (Center 90°)
    Tilt (Pin 33): 45° ~ 135° (Center 90°)
    """

    def __init__(self,
                 pan_min=0.0, pan_max=180.0, pan_start=90.0,
                 tilt_min=45.0, tilt_max=135.0, tilt_start=90.0,
                 speed_dps=180.0, accel_dps2=800.0, rate_hz=100):
        fix_jetson_hardware_registers()

        chip_path = "/sys/class/pwm/pwmchip0"
        self.pan_hw = SysfsPWMChannel(chip_path, channel=0)   # Pin 32 -> PWM0
        self.tilt_hw = SysfsPWMChannel(chip_path, channel=2)  # Pin 33 -> PWM2

        self.pan_axis = _AxisMotion("Pan", pan_start, pan_min, pan_max, speed_dps, accel_dps2)
        self.tilt_axis = _AxisMotion("Tilt", tilt_start, tilt_min, tilt_max, speed_dps, accel_dps2)

        self.lock = threading.Lock()
        self.period = 1.0 / rate_hz
        self.running = True

        # Initial write
        self.pan_hw.set_angle(self.pan_axis.pos)
        self.tilt_hw.set_angle(self.tilt_axis.pos)

        # Recent angle history for camera timestamp alignment
        self.recent = deque(maxlen=int(rate_hz * 2))

        self.thread = threading.Thread(target=self._motion_loop, daemon=True)
        self.thread.start()

    def _motion_loop(self):
        last_time = time.monotonic()
        while self.running:
            time.sleep(self.period)
            now = time.monotonic()
            dt = min(0.05, now - last_time)
            last_time = now

            with self.lock:
                pan_pos = self.pan_axis.step(dt)
                tilt_pos = self.tilt_axis.step(dt)
                self.recent.append((now, pan_pos, tilt_pos))

            self.pan_hw.set_angle(pan_pos)
            self.tilt_hw.set_angle(tilt_pos)

    def set_target(self, pan_deg: float, tilt_deg: float, speed: float = None):
        """Set absolute target angles in degrees."""
        with self.lock:
            self.pan_axis.set_target(pan_deg, speed)
            self.tilt_axis.set_target(tilt_deg, speed)

    def jog(self, pan_dir: float, tilt_dir: float, speed_dps: float = 60.0):
        """
        Jogging control for manual mode.
        pan_dir: -1 (left), 0 (none), +1 (right)
        tilt_dir: -1 (down), 0 (none), +1 (up)
        """
        with self.lock:
            # Advance target smoothly based on jogging direction
            if pan_dir != 0:
                step = pan_dir * speed_dps * 0.1
                new_pan = max(self.pan_axis.min_deg, min(self.pan_axis.max_deg, self.pan_axis.target + step))
                self.pan_axis.set_target(new_pan, speed_dps)
            else:
                self.pan_axis.set_target(self.pan_axis.pos, speed_dps)

            if tilt_dir != 0:
                step = tilt_dir * speed_dps * 0.1
                new_tilt = max(self.tilt_axis.min_deg, min(self.tilt_axis.max_deg, self.tilt_axis.target + step))
                self.tilt_axis.set_target(new_tilt, speed_dps)
            else:
                self.tilt_axis.set_target(self.tilt_axis.pos, speed_dps)

    def center(self, speed_dps: float = 80.0):
        """Smoothly returns both axes to 90.0° center."""
        self.set_target(90.0, 90.0, speed_dps)

    def get_position(self):
        """Returns (current_pan, current_tilt, target_pan, target_tilt)."""
        with self.lock:
            return (self.pan_axis.pos, self.tilt_axis.pos,
                    self.pan_axis.target, self.tilt_axis.target)

    def position_at(self, when: float):
        """Returns the interpolated (pan, tilt) angle at a specific monotonic timestamp."""
        with self.lock:
            if not self.recent or when is None:
                return self.pan_axis.pos, self.tilt_axis.pos
            samples = list(self.recent)

        if when <= samples[0][0]:
            return samples[0][1], samples[0][2]
        for (t1, p1, q1), (t2, p2, q2) in zip(samples, samples[1:]):
            if t1 <= when <= t2:
                k = (when - t1) / (t2 - t1) if t2 > t1 else 0.0
                return p1 + k * (p2 - p1), q1 + k * (q2 - q1)
        return samples[-1][1], samples[-1][2]

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.pan_hw.release()
        self.tilt_hw.release()
