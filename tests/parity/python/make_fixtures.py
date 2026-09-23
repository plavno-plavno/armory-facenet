"""Build the parity fixture set (>= 50 images: varied people, light, pose; 0/1/many faces)."""
import os, shutil, sys
import numpy as np, cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, 'fixtures', 'raw')
OUT = os.path.join(ROOT, 'fixtures', 'parity')
os.makedirs(OUT, exist_ok=True)

lfw = os.path.join(RAW, 'lfw')
people = sorted(os.listdir(lfw))
picked = 0
for i, p in enumerate(people[::97]):          # deterministic spread over the alphabet
    imgs = sorted(os.listdir(os.path.join(lfw, p)))
    shutil.copy(os.path.join(lfw, p, imgs[0]), os.path.join(OUT, f'lfw_{imgs[0]}'))
    picked += 1
    if picked >= 56:
        break
for f in ['t1.jpg', 'messi5.jpg', 'lena.jpg', 'largest_selfie.jpg']:
    shutil.copy(os.path.join(RAW, f), os.path.join(OUT, f))
# Large frame where the face is bigger than YuNet's ~300px range until downscaled.
big = cv2.imread(os.path.join(RAW, 'lena.jpg'))
cv2.imwrite(os.path.join(OUT, 'lena_x3.jpg'), cv2.resize(big, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC))
# No-face images.
rng = np.random.default_rng(0)
cv2.imwrite(os.path.join(OUT, 'noface_noise.png'), rng.integers(0, 255, (360, 480, 3), dtype=np.uint8))
grad = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
cv2.imwrite(os.path.join(OUT, 'noface_gradient.png'), cv2.merge([grad, grad[::-1], np.full_like(grad, 90)]))
print(len(os.listdir(OUT)), 'fixtures in', OUT)
