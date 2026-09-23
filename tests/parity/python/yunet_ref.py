"""Reference detection: cv2.FaceDetectorYN on the same model, same downscale rule as the engine."""
import math
import cv2
import numpy as np

def round_half_up(x):
    return int(math.floor(x + 0.5))

def detect(model_path, img, long_side=640, score=0.9, nms=0.3):
    h, w = img.shape[:2]
    s = min(1.0, long_side / max(w, h))
    if s < 1.0:
        rw, rh = max(1, round_half_up(w * s)), max(1, round_half_up(h * s))
        img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    else:
        rw, rh = w, h
    det = cv2.FaceDetectorYN.create(model_path, '', (rw, rh), score, nms, 5000)
    _, faces = det.detect(img)
    out = []
    if faces is None:
        return out
    fx, fy = w / rw, h / rh
    for f in faces:
        out.append({
            'box': [float(f[0] * fx), float(f[1] * fy), float(f[2] * fx), float(f[3] * fy)],
            'landmarks': [[float(f[4 + 2 * k] * fx), float(f[5 + 2 * k] * fy)] for k in range(5)],
            'score': float(f[14]),
        })
    return out
