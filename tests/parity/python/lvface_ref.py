"""Reference embeddings via LVFaceONNXInferencer (inference_onnx.py from the LVFace repo, unmodified)."""
import numpy as np
from lvface_inference_onnx import LVFaceONNXInferencer

class Ref:
    def __init__(self, model_path):
        self.inf = LVFaceONNXInferencer(model_path, use_gpu=False)

    def embed_bgr(self, aligned_bgr):
        t = self.inf._preprocess_image(aligned_bgr)
        out = self.inf.ort_session.run([self.inf.output_name], {self.inf.input_name: t})[0][0]
        return (out / np.linalg.norm(out)).astype(np.float32)
