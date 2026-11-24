#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import time
import sys
import os

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.servo_jetson import ServoMover, _AxisMotion


class TestServoLimits(unittest.TestCase):
    def test_axis_limits(self):
        # Pan: 0 to 180
        pan = _AxisMotion("Pan", start_deg=90.0, min_deg=0.0, max_deg=180.0)
        pan.set_target(-50.0)
        self.assertEqual(pan.target, 0.0)
        pan.set_target(250.0)
        self.assertEqual(pan.target, 180.0)

        # Tilt: 45 to 135
        tilt = _AxisMotion("Tilt", start_deg=90.0, min_deg=45.0, max_deg=135.0)
        tilt.set_target(10.0)
        self.assertEqual(tilt.target, 45.0)
        tilt.set_target(170.0)
        self.assertEqual(tilt.target, 135.0)

    def test_servo_mover_smooth_gliding(self):
        mover = ServoMover(pan_min=0.0, pan_max=180.0, pan_start=90.0,
                           tilt_min=45.0, tilt_max=135.0, tilt_start=90.0,
                           speed_dps=500.0, accel_dps2=2000.0)
        try:
            mover.set_target(120.0, 70.0)
            time.sleep(0.3)
            pan_pos, tilt_pos, pan_tgt, tilt_tgt = mover.get_position()
            self.assertGreater(pan_pos, 90.0)
            self.assertLess(tilt_pos, 90.0)
            self.assertLessEqual(pan_pos, 180.0)
            self.assertGreaterEqual(tilt_pos, 45.0)
        finally:
            mover.stop()


if __name__ == "__main__":
    unittest.main()
