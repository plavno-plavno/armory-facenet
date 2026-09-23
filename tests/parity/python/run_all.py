"""Run all reference stages over tests/fixtures/parity and write tests/parity/expected/."""
import json, os, sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from yunet_ref import detect
from align_ref import norm_crop
from lvface_ref import Ref

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
FIX = os.path.join(ROOT, 'tests', 'fixtures', 'parity')
EXP = os.path.join(ROOT, 'tests', 'parity', 'expected')
MODELS = os.path.join(ROOT, 'models')
manifest = json.load(open(os.path.join(MODELS, 'manifest.json')))
embedders = [manifest['embedder']] + manifest.get('embedders', [])

os.makedirs(os.path.join(EXP, 'aligned'), exist_ok=True)
refs = {e['id']: Ref(os.path.join(MODELS, e['file'])) for e in embedders if os.path.exists(os.path.join(MODELS, e['file']))}

result = {'cv2': cv2.__version__, 'images': []}
for name in sorted(os.listdir(FIX)):
    img = cv2.imread(os.path.join(FIX, name), cv2.IMREAD_COLOR)
    faces = detect(os.path.join(MODELS, manifest['detector']['file']), img)
    for i, f in enumerate(faces):
        crop, M = norm_crop(img, f['landmarks'])
        crop_name = f'{name}.{i}.png'
        cv2.imwrite(os.path.join(EXP, 'aligned', crop_name), crop)
        f['aligned'] = crop_name
        f['transform'] = M.flatten().tolist()
        f['embeddings'] = {k: r.embed_bgr(crop).tolist() for k, r in refs.items()}
    result['images'].append({'file': name, 'width': img.shape[1], 'height': img.shape[0], 'faces': faces})
    print(name, len(faces))

with open(os.path.join(EXP, 'reference.json'), 'w') as fp:
    json.dump(result, fp)
print('faces total', sum(len(i['faces']) for i in result['images']))
