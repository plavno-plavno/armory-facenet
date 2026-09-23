"""Reference liveness (anti-spoofing): MiniFASNet as in minivision-ai/Silent-Face-Anti-Spoofing.

Crop: CropImage._get_new_box (box scaled around its center, shifted - not clipped - into the image),
cv2.resize INTER_LINEAR to the model input, BGR, raw 0..255 float, NCHW. Output: softmax over
3 classes, class 1 = real. Several models are averaged (test.py sums predictions over models).
"""
import cv2
import numpy as np
import onnxruntime as ort


def new_box(src_w, src_h, bbox, scale):
    x, y, box_w, box_h = bbox
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def crop(img, bbox, scale, out_w, out_h):
    src_h, src_w = img.shape[:2]
    x1, y1, x2, y2 = new_box(src_w, src_h, bbox, scale)
    return cv2.resize(img[y1:y2 + 1, x1:x2 + 1], (out_w, out_h))


class Ref:
    def __init__(self, path, scale, size):
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        self.scale = scale
        self.size = size

    def real_prob(self, img, box):
        # Detector boxes are integers in the original pipeline.
        bbox = [int(v) for v in box]
        c = crop(img, bbox, self.scale, self.size[1], self.size[0]).astype(np.float32)
        x = np.transpose(c, (2, 0, 1))[None]
        logits = self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return float((e / e.sum(axis=1, keepdims=True))[0, 1])
