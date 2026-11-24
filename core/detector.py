#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ultra-Light-Fast-Generic-Face-Detector-1MB wrapper for Jetson Nano.
Uses OpenCV DNN with CUDA backend for sub-10ms inference.
Implements target persistence (nearest-neighbor centroid tracking) to prevent target-hopping.
"""

import os
import sys
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


class UltraLightFaceDetector:
    """1MB Ultra-lightweight Face Detector using OpenCV DNN."""

    def __init__(self, model_path: str = None, conf_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.net = None
        self.use_cuda = False
        self.last_target_center = None

        if model_path is None:
            # Look in standard submodule location
            here = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(here, "..", "modules", "Ultra-Light-Fast-Generic-Face-Detector-1MB", "models", "onnx", "version-RFB-320.onnx"),
                os.path.join(here, "..", "modules", "Ultra-Light-Fast-Generic-Face-Detector-1MB", "models", "onnx", "version-slim-320.onnx"),
                os.path.join(here, "..", "models", "version-RFB-320.onnx"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    model_path = os.path.abspath(c)
                    break

        if cv2 is not None and model_path and os.path.exists(model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(model_path)
                # Try setting CUDA backend on Jetson
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    self.use_cuda = True
                except Exception:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    self.use_cuda = False
                print(f"[Detector] Loaded 1MB model: {model_path} (CUDA: {self.use_cuda})")
            except Exception as e:
                print(f"[Detector] Failed to load ONNX via cv2.dnn: {e}")
                self.net = None
        else:
            print(f"[Detector] ONNX model not found at {model_path}, fallback mode.")

    def detect(self, frame: np.ndarray):
        """
        Detects faces in frame.
        Returns: list of dicts:
            [{ 'box': (x, y, w, h), 'score': float, 'center': (cx, cy), 'area': int }]
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        if self.net is None:
            # Fallback: check if frame is synthetic or use simple skin color/brightest spot
            return []

        # Ultra-Light 1MB expects 320x240 RGB normalized
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0 / 128.0,
            size=(320, 240),
            mean=(127.0, 127.0, 127.0),
            swapRB=True,
            crop=False
        )

        self.net.setInput(blob)
        out_names = self.net.getUnconnectedOutLayersNames()
        outputs = self.net.forward(out_names)

        # Output shape: scores (1, num_anchors, 2), boxes (1, num_anchors, 4)
        scores_arr = None
        boxes_arr = None
        for out in outputs:
            if out.shape[-1] == 2:
                scores_arr = out[0]
            elif out.shape[-1] == 4:
                boxes_arr = out[0]

        if scores_arr is None or boxes_arr is None:
            return []

        # Class 1 is face confidence
        confidences = scores_arr[:, 1]
        mask = confidences > self.conf_threshold

        cand_boxes = boxes_arr[mask]
        cand_scores = confidences[mask]

        if len(cand_scores) == 0:
            return []

        # Convert normalized coordinates [x1, y1, x2, y2] to absolute [x, y, w, h]
        rects = []
        scores = []
        for i in range(len(cand_scores)):
            b = cand_boxes[i]
            x1 = int(max(0.0, b[0] * w))
            y1 = int(max(0.0, b[1] * h))
            x2 = int(min(float(w), b[2] * w))
            y2 = int(min(float(h), b[3] * h))
            bw = x2 - x1
            bh = y2 - y1
            if bw > 10 and bh > 10:
                rects.append([x1, y1, bw, bh])
                scores.append(float(cand_scores[i]))

        if not rects:
            return []

        indices = cv2.dnn.NMSBoxes(rects, scores, self.conf_threshold, self.nms_threshold)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                rx, ry, rw, rh = rects[idx]
                score = scores[idx]
                cx = rx + rw // 2
                cy = ry + rh // 2
                area = rw * rh
                results.append({
                    'box': (rx, ry, rw, rh),
                    'score': score,
                    'center': (cx, cy),
                    'area': area
                })

        return results

    def select_primary_face(self, faces: list):
        """
        Selects the best face to track:
          - If already tracking a face, picks the one closest to last_target_center (if within 100px);
          - Otherwise picks the largest face by bounding box area.
        """
        if not faces:
            self.last_target_center = None
            return None

        if self.last_target_center is not None:
            # Find closest to previous center
            px, py = self.last_target_center
            closest = min(faces, key=lambda f: (f['center'][0] - px)**2 + (f['center'][1] - py)**2)
            dist = np.sqrt((closest['center'][0] - px)**2 + (closest['center'][1] - py)**2)
            if dist < 120.0:  # Consistent target
                self.last_target_center = closest['center']
                return closest

        # Pick largest
        best = max(faces, key=lambda f: f['area'])
        self.last_target_center = best['center']
        return best
