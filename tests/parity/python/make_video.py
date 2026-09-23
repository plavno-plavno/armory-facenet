"""Synthetic stream fixture: an enrolled person, a gap, a stranger, a gap (1280x720, 25 fps).
Also writes enrollment photos (x2 upscaled LFW images, disjoint from the video frames)."""
import os
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LFW = os.path.join(ROOT, 'tests', 'fixtures', 'raw', 'lfw')
OUT = os.path.join(ROOT, 'tests', 'fixtures')
W, H, FPS = 1280, 720, 25
os.makedirs(os.path.join(OUT, 'enroll'), exist_ok=True)
os.makedirs(os.path.join(OUT, 'video'), exist_ok=True)


def load(person, idx, scale=2):
    img = cv2.imread(os.path.join(LFW, person, f'{person}_{idx:04d}.jpg'))
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def place(img):
    canvas = np.full((H, W, 3), 40, np.uint8)
    h, w = img.shape[:2]
    y, x = (H - h) // 2, (W - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


# Enrollment photos (not used in the video).
for person, idxs in {'George_W_Bush': [1, 2, 5], 'Colin_Powell': [1, 2, 3], 'Tony_Blair': [1, 2]}.items():
    for i in idxs:
        cv2.imwrite(os.path.join(OUT, 'enroll', f'{person}_{i:04d}.jpg'), load(person, i), [cv2.IMWRITE_JPEG_QUALITY, 95])

segments = [
    ('George_W_Bush', [10, 11, 12, 13, 14]),
    (None, 2.0),
    ('Gerhard_Schroeder', [1, 2, 3, 4, 5]),  # not enrolled
    (None, 2.0),
]
vw = cv2.VideoWriter(os.path.join(OUT, 'video', 'enrolled_then_stranger.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
for who, arg in segments:
    if who is None:
        for _ in range(int(arg * FPS)):
            vw.write(np.full((H, W, 3), 40, np.uint8))
        continue
    for idx in arg:
        frame = place(load(who, idx))
        for _ in range(int(0.6 * FPS)):
            vw.write(frame)
vw.release()
print('ok')
