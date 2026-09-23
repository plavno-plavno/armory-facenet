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

# Liveness (anti-spoofing): per-model real-class probability on the reference detector boxes.
from liveness_ref import Ref as LivenessRef
live_models = [m for m in manifest.get('liveness', []) if os.path.exists(os.path.join(MODELS, m['file']))]
if live_models:
    lrefs = {m['id']: LivenessRef(os.path.join(MODELS, m['file']), m['cropScale'], tuple(m['inputSize'])) for m in live_models}
    live = {'images': []}
    for d in [FIX, os.path.join(ROOT, 'tests', 'fixtures', 'liveness')]:
        for name in sorted(os.listdir(d)):
            img = cv2.imread(os.path.join(d, name), cv2.IMREAD_COLOR)
            faces = detect(os.path.join(MODELS, manifest['detector']['file']), img)
            live['images'].append({
                'file': os.path.relpath(os.path.join(d, name), os.path.join(ROOT, 'tests', 'fixtures')),
                'faces': [{'box': f['box'], 'real': {k: r.real_prob(img, f['box']) for k, r in lrefs.items()}} for f in faces],
            })
    with open(os.path.join(EXP, 'liveness.json'), 'w') as fp:
        json.dump(live, fp, indent=1)
    print('liveness faces', sum(len(i['faces']) for i in live['images']))
