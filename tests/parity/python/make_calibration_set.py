"""Demo calibration dataset from LFW (for tooling tests only; real calibration needs target-camera footage).

usage: make_calibration_set.py <out_dir> [persons=40]
Each person: enroll/ = 3 images, passes/pNN/ = 1 other image as a 3-frame pass (slight shifts).
"""
import os
import sys
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LFW = os.path.join(ROOT, 'tests', 'fixtures', 'raw', 'lfw')
out = sys.argv[1]
n_persons = int(sys.argv[2]) if len(sys.argv) > 2 else 40

people = sorted(p for p in os.listdir(LFW) if len(os.listdir(os.path.join(LFW, p))) >= 14)[:n_persons]
groups = ['group,A', 'group,B']
with open(os.path.join(out, 'groups.csv') if os.makedirs(out, exist_ok=True) is None else '', 'w') as g:
    g.write('personId,group\n')
    for i, p in enumerate(people):
        g.write(f'{p},{"A" if i % 2 else "B"}\n')

def up(img):
    return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

for p in people:
    files = sorted(os.listdir(os.path.join(LFW, p)))
    os.makedirs(os.path.join(out, p, 'enroll'), exist_ok=True)
    for f in files[:3]:
        cv2.imwrite(os.path.join(out, p, 'enroll', f), up(cv2.imread(os.path.join(LFW, p, f))))
    for k, f in enumerate(files[3:13]):
        d = os.path.join(out, p, 'passes', f'p{k:02d}')
        os.makedirs(d, exist_ok=True)
        img = up(cv2.imread(os.path.join(LFW, p, f)))
        for j, (dx, dy) in enumerate([(0, 0), (4, 2), (-3, 3)]):
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            cv2.imwrite(os.path.join(d, f'{j}.jpg'), cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REPLICATE))
print(len(people), 'persons ->', out)
