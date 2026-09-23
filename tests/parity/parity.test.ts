// Parity tests Node vs Python reference (ТЗ §15.1). Reference data: npm run parity:ref.
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { beforeAll, describe, expect, it } from 'vitest';
import { loadVision, type VisionStack } from '../../apps/engine/src/vision/index.js';
import { decodeImage } from '../../apps/engine/src/vision/image.js';
import { dot } from '../../apps/engine/src/vision/embedder.js';
import { rectIoU } from '../../apps/engine/src/vision/yunet.js';
import type { AlignedFace, Detection, Landmarks5 } from '../../apps/engine/src/vision/types.js';
import { OnnxEmbedder } from '../../apps/engine/src/vision/embedder.js';
import { createSession, loadManifest } from '../../apps/engine/src/vision/models.js';

const ROOT = path.resolve(__dirname, '../..');
const FIX = path.join(ROOT, 'tests/fixtures/parity');
const EXP = path.join(ROOT, 'tests/parity/expected');
const MODELS = path.join(ROOT, 'models');

interface RefFace extends Detection {
  aligned: string;
  transform: number[];
  embeddings: Record<string, number[]>;
}
interface RefImage {
  file: string;
  width: number;
  height: number;
  faces: RefFace[];
}

const ref: { images: RefImage[] } = JSON.parse(readFileSync(path.join(EXP, 'reference.json'), 'utf8'));
let vision: VisionStack;

async function loadAligned(name: string): Promise<AlignedFace> {
  const { data, info } = await sharp(path.join(EXP, 'aligned', name)).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  expect(info.width).toBe(112);
  return { rgb: new Uint8Array(data), size: 112, transform: [] };
}

/** Greedy match of detections by IoU. */
function pair(ours: Detection[], theirs: Detection[]) {
  const used = new Set<number>();
  return theirs.map((t) => {
    let best = -1;
    let bestIoU = 0;
    ours.forEach((o, i) => {
      if (used.has(i)) return;
      const v = rectIoU(o.box, t.box);
      if (v > bestIoU) {
        bestIoU = v;
        best = i;
      }
    });
    if (best >= 0) used.add(best);
    return { ref: t, ours: best >= 0 ? ours[best] : undefined, iou: bestIoU };
  });
}

beforeAll(async () => {
  vision = await loadVision({
    modelsDir: MODELS,
    executionProvider: 'cpu',
    intraOpThreads: 4,
    detector: { inputLongSide: 640, scoreThreshold: 0.9, nmsThreshold: 0.3 },
  });
});

describe('parity: detection (cv2.FaceDetectorYN)', () => {
  it(`fixture set has >= 50 images`, () => {
    expect(ref.images.length).toBeGreaterThanOrEqual(50);
  });

  const stats = { faces: 0, lmErrSum: 0, lmCount: 0 };
  for (const img of ref.images) {
    it(img.file, async () => {
      const dec = await decodeImage(readFileSync(path.join(FIX, img.file)));
      expect([dec.frame.width, dec.frame.height]).toEqual([img.width, img.height]);
      const dets = await vision.detector.detect(dec.frame);
      expect(dets.length).toBe(img.faces.length);
      for (const p of pair(dets, img.faces)) {
        expect(p.iou).toBeGreaterThanOrEqual(0.95);
        expect(Math.abs(p.ours!.score - p.ref.score)).toBeLessThanOrEqual(0.01);
        let err = 0;
        for (let k = 0; k < 5; k++) err += Math.hypot(p.ours!.landmarks[k][0] - p.ref.landmarks[k][0], p.ours!.landmarks[k][1] - p.ref.landmarks[k][1]);
        stats.lmErrSum += err;
        stats.lmCount += 5;
        stats.faces++;
      }
    });
  }
  it('mean landmark error <= 1 px', () => {
    const mean = stats.lmErrSum / stats.lmCount;
    console.log(`[parity] detection: ${stats.faces} faces, mean landmark error ${mean.toFixed(3)} px`);
    expect(mean).toBeLessThanOrEqual(1);
  });
});

describe('parity: alignment (insightface norm_crop)', () => {
  const faces = ref.images.flatMap((i) => i.faces.map((f) => ({ img: i, f })));
  let worst = 0;
  for (const { img, f } of faces) {
    it(f.aligned, async () => {
      const dec = await decodeImage(readFileSync(path.join(FIX, img.file)));
      const ours = vision.aligner.align(dec.frame, f.landmarks as Landmarks5);
      // Reference uses float32 landmarks/template: compare with a relative tolerance.
      for (let i = 0; i < 6; i++) expect(Math.abs(ours.transform[i] - f.transform[i])).toBeLessThanOrEqual(1e-5 * Math.max(1, Math.abs(f.transform[i])));
      const theirs = await loadAligned(f.aligned);
      let diff = 0;
      for (let i = 0; i < ours.rgb.length; i++) diff += Math.abs(ours.rgb[i] - theirs.rgb[i]);
      const mad = diff / ours.rgb.length;
      worst = Math.max(worst, mad);
      expect(mad).toBeLessThanOrEqual(1.0);
    });
  }
  it('report', () => console.log(`[parity] alignment: worst mean abs diff ${worst.toFixed(3)}`));
});

describe('parity: embeddings (LVFace inference_onnx.py)', () => {
  const manifest = loadManifest(MODELS);
  const all = [manifest.embedder, ...(manifest.embedders ?? [])].filter((e) => existsSync(path.join(MODELS, e.file)));
  const faces = ref.images.flatMap((i) => i.faces);
  for (const em of all) {
    it(`${em.id}: cosine >= 0.999 on every face`, async () => {
      if (!faces[0].embeddings[em.id]) return; // reference not generated for this model
      const { session } = await createSession(path.join(MODELS, em.file), 'cpu', 4);
      const embedder = new OnnxEmbedder(session, em);
      const crops = await Promise.all(faces.map((f) => loadAligned(f.aligned)));
      let min = 1;
      for (let i = 0; i < crops.length; i += 8) {
        const embs = await embedder.embed(crops.slice(i, i + 8));
        embs.forEach((e, j) => {
          const cos = dot(e, Float32Array.from(faces[i + j].embeddings[em.id]));
          min = Math.min(min, cos);
        });
      }
      console.log(`[parity] ${em.id}: min cosine ${min.toFixed(6)} over ${faces.length} faces`);
      expect(min).toBeGreaterThanOrEqual(0.999);
    });
  }
});

describe('end-to-end: Node pipeline vs Python pipeline', () => {
  // Not a §15.1 criterion (stages are checked separately above); guards against gross drift.
  it('embedding of full chain (decode→detect→align→embed) agrees', async () => {
    const id = vision.embedderManifest.id;
    let min = 1;
    for (const img of ref.images.filter((i) => i.faces.length > 0)) {
      const dec = await decodeImage(readFileSync(path.join(FIX, img.file)));
      const dets = await vision.detector.detect(dec.frame);
      for (const p of pair(dets, img.faces)) {
        const [e] = await vision.embedder.embed([vision.aligner.align(dec.frame, p.ours!.landmarks)]);
        min = Math.min(min, dot(e, Float32Array.from(p.ref.embeddings[id])));
      }
    }
    console.log(`[parity] end-to-end min cosine ${min.toFixed(5)}`);
    expect(min).toBeGreaterThanOrEqual(0.98);
  });
});
