"""Reference alignment: verbatim insightface.utils.face_align.norm_crop (arcface mode, 112)."""
import cv2
import numpy as np
from skimage import transform as trans

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112):
    dst = arcface_dst * (float(image_size) / 112.0)
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    return tform.params[0:2, :]

def norm_crop(img, landmark, image_size=112):
    M = estimate_norm(np.asarray(landmark, dtype=np.float32), image_size)
    return cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0), M
