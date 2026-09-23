import { readFileSync } from 'node:fs';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';
import { ARCFACE_TEMPLATE, estimateSimilarity, invertAffine, warpAffine } from '../../src/vision/align.js';
import { decodeImage } from '../../src/vision/image.js';
import { loadVision, type VisionStack } from '../../src/vision/index.js';
import { buildDetectorInput, resizeBilinear } from '../../src/vision/preprocess.js';
import { DefaultQualityAssessor, laplacianVariance } from '../../src/vision/quality.js';
import type { Detection, Landmarks5, Point } from '../../src/vision/types.js';
import { decodeYuNet, nms, rectIoU } from '../../src/vision/yunet.js';
import { FIX, MODELS } from '../helpers.js';

describe('YuNet decoding', () => {
  it('decodes a single anchor exactly as FaceDetectorYN', () => {
    const padW = 64;
    const padH = 64;
    const mk = (s: number, per: number) => new Float32Array((padW / s) * (padH / s) * per);
    const out = { cls: [mk(8, 1), mk(16, 1), mk(32, 1)], obj: [mk(8, 1), mk(16, 1), mk(32, 1)], bbox: [mk(8, 4), mk(16, 4), mk(32, 4)], kps: [mk(8, 10), mk(16, 10), mk(32, 10)] };
    // stride 16, row 1, col 2 -> index 1*4+2 = 6
    const i = 6;
    out.cls[1][i] = 1.2; // clamped to 1
    out.obj[1][i] = 0.81;
    out.bbox[1].set([0.5, 0.25, Math.log(2), Math.log(3)], i * 4);
    out.kps[1].set([0.1, 0.2, 0.9, 0.2, 0.5, 0.5, 0.2, 0.8, 0.8, 0.8], i * 10);
    const dets = decodeYuNet(out, padW, padH, 0.5);
    expect(dets).toHaveLength(1);
    const d = dets[0];
    expect(d.score).toBeCloseTo(0.9, 6); // sqrt(1 * 0.81)
    const cx = (2 + 0.5) * 16;
    const cy = (1 + 0.25) * 16;
    // inputs are float32, so compare with float32 tolerance
    [cx - 16, cy - 24, 32, 48].forEach((v, k) => expect(d.box[k]).toBeCloseTo(v, 4));
    expect(d.landmarks[0][0]).toBeCloseTo((0.1 + 2) * 16, 4);
    expect(d.landmarks[0][1]).toBeCloseTo((0.2 + 1) * 16, 4);
    expect(d.landmarks[4][1]).toBeCloseTo((0.8 + 1) * 16);
  });

  it('filters by score threshold', () => {
    const z = (n: number) => new Float32Array(n);
    const out = { cls: [z(64), z(16), z(4)], obj: [z(64), z(16), z(4)], bbox: [z(256), z(64), z(16)], kps: [z(640), z(160), z(40)] };
    out.cls[0][0] = 0.5;
    out.obj[0][0] = 0.5;
    expect(decodeYuNet(out, 64, 64, 0.9)).toHaveLength(0);
    expect(decodeYuNet(out, 64, 64, 0.5)).toHaveLength(1);
  });
});

describe('NMS', () => {
  const det = (box: [number, number, number, number], score: number): Detection => ({ box, score, landmarks: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] });

  it('suppresses overlapping lower-score boxes and keeps disjoint ones', () => {
    const kept = nms([det([0, 0, 100, 100], 0.95), det([5, 5, 100, 100], 0.97), det([300, 300, 50, 50], 0.92)], 0.9, 0.3);
    expect(kept.map((d) => d.score)).toEqual([0.97, 0.92]);
  });

  it('uses integer-truncated rects and strict score filter like cv::dnn::NMSBoxes', () => {
    expect(nms([det([0, 0, 10, 10], 0.9)], 0.9, 0.3)).toHaveLength(0);
    expect(rectIoU([0, 0, 10, 10], [5, 0, 10, 10])).toBeCloseTo(50 / 150);
  });
});

describe('alignment', () => {
  it('recovers a known similarity transform (Umeyama)', () => {
    const angle = 0.3;
    const s = 1.7;
    const tx = 12;
    const ty = -5;
    const src: Point[] = [[10, 20], [60, 25], [35, 50], [15, 70], [55, 72]];
    const dst = src.map(([x, y]) => [s * (Math.cos(angle) * x - Math.sin(angle) * y) + tx, s * (Math.sin(angle) * x + Math.cos(angle) * y) + ty] as Point);
    const m = estimateSimilarity(src, dst);
    expect(m[0]).toBeCloseTo(s * Math.cos(angle), 9);
    expect(m[3]).toBeCloseTo(s * Math.sin(angle), 9);
    expect(m[2]).toBeCloseTo(tx, 9);
    expect(m[5]).toBeCloseTo(ty, 9);
  });

  it('never produces a reflection', () => {
    const mirrored = ARCFACE_TEMPLATE.map(([x, y]) => [112 - x, y] as Point);
    const m = estimateSimilarity(mirrored, ARCFACE_TEMPLATE);
    expect(m[0] * m[4] - m[1] * m[3]).toBeGreaterThan(0);
  });

  it('invertAffine is an inverse', () => {
    const m = [1.2, -0.3, 5, 0.3, 1.2, -7];
    const inv = invertAffine(m);
    const x = 13;
    const y = 29;
    const u = m[0] * x + m[1] * y + m[2];
    const v = m[3] * x + m[4] * y + m[5];
    expect(inv[0] * u + inv[1] * v + inv[2]).toBeCloseTo(x, 9);
    expect(inv[3] * u + inv[4] * v + inv[5]).toBeCloseTo(y, 9);
  });

  it('identity warp copies pixels; out-of-frame is 0', () => {
    const src = new Uint8Array(4 * 4 * 3).map((_, i) => i);
    expect(warpAffine(src, 4, 4, [1, 0, 0, 0, 1, 0], 4, 4)).toEqual(src);
    const shifted = warpAffine(src, 4, 4, [1, 0, 10, 0, 1, 0], 4, 4);
    expect([...shifted].every((v) => v === 0)).toBe(true);
  });
});

describe('preprocessing', () => {
  it('bilinear resize of a constant image stays constant; detector input pads to 32', () => {
    const src = new Uint8Array(100 * 50 * 3).fill(77);
    expect([...resizeBilinear(src, 100, 50, 33, 17)].every((v) => v === 77)).toBe(true);
    const inp = buildDetectorInput({ data: src, width: 100, height: 50, ts: 0, sourceId: 'x' }, 640);
    expect([inp.padW, inp.padH]).toEqual([128, 64]);
    expect(inp.scaleX).toBe(1);
    expect(inp.tensor[128 * 64 - 1]).toBe(0); // padding
    const big = buildDetectorInput({ data: new Uint8Array(1920 * 1080 * 3), width: 1920, height: 1080, ts: 0, sourceId: 'x' }, 640);
    expect([big.resizedW, big.resizedH, big.padW, big.padH]).toEqual([640, 360, 640, 384]);
  });
});

describe('quality metrics', () => {
  it('laplacian variance: flat = 0, checkerboard high', () => {
    expect(laplacianVariance(new Float32Array(16 * 16).fill(100), 16, 16)).toBe(0);
    const cb = new Float32Array(16 * 16).map((_, i) => (((i % 16) + Math.floor(i / 16)) % 2 ? 255 : 0));
    expect(laplacianVariance(cb, 16, 16)).toBeGreaterThan(10000);
  });

  it('flags yaw, roll, size and out-of-frame', () => {
    const q = new DefaultQualityAssessor({
      minDetScore: 0.9, minFaceSize: 80, minInterocular: 30, maxYaw: 0.25, maxRollDeg: 25, minSharpness: 0,
      brightness: [0, 255], requireInFrame: true, weights: { detScore: 1, faceSize: 1, yaw: 1, sharpness: 1 },
    });
    const frame = { data: new Uint8Array(3), width: 200, height: 200, ts: 0, sourceId: 'x' };
    const face = { rgb: new Uint8Array(112 * 112 * 3).fill(128), size: 112 as const, transform: [] };
    const good: Detection = { box: [50, 50, 100, 100], score: 0.95, landmarks: [[80, 90], [120, 90], [100, 110], [85, 130], [115, 130]] };
    expect(q.assess(frame, good, face).reasons).toEqual([]);
    const turned: Detection = { ...good, landmarks: [[80, 90], [120, 90], [115, 110], [85, 130], [115, 130]] };
    expect(q.assess(frame, turned, face).reasons).toContain('yaw');
    const rolled: Detection = { ...good, landmarks: [[80, 80], [120, 110], [100, 110], [85, 130], [115, 130]] };
    expect(q.assess(frame, rolled, face).reasons).toContain('roll');
    const edge: Detection = { ...good, box: [150, 50, 100, 100] };
    expect(q.assess(frame, edge, face).reasons).toContain('inFrame');
    const small: Detection = { ...good, box: [50, 50, 40, 40] };
    expect(q.assess(frame, small, face).reasons).toContain('faceSize');
  });
});

describe('landmark order (spec §6.1: mandatory)', () => {
  let v: VisionStack;
  beforeAll(async () => {
    v = await loadVision({ modelsDir: MODELS, executionProvider: 'cpu', detector: { inputLongSide: 640, scoreThreshold: 0.9, nmsThreshold: 0.3 } });
  });

  it('frontal face: landmark[0].x < landmark[1].x (right eye of the person is on the image left)', async () => {
    for (const f of ['lena.jpg', 'George_W_Bush_0002.jpg']) {
      const p = f === 'lena.jpg' ? path.join(FIX, 'raw', f) : path.join(FIX, 'enroll', f);
      const { frame } = await decodeImage(readFileSync(p));
      const [d] = await v.detector.detect(frame);
      const lm = d.landmarks as Landmarks5;
      expect(lm[0][0]).toBeLessThan(lm[1][0]);
      expect(lm[3][0]).toBeLessThan(lm[4][0]);
      expect(lm[2][1]).toBeGreaterThan(lm[0][1]); // nose below eyes
    }
  });

  it('large faces (> ~300px) are found thanks to downscaling (R10)', async () => {
    const { frame } = await decodeImage(readFileSync(path.join(FIX, 'parity', 'lena_x3.jpg')));
    const dets = await v.detector.detect(frame);
    expect(dets.length).toBeGreaterThanOrEqual(1);
    expect(dets[0].box[2]).toBeGreaterThan(300);
  });

  it('embeddings are L2-normalized and batch == single', async () => {
    const { frame } = await decodeImage(readFileSync(path.join(FIX, 'enroll', 'Tony_Blair_0001.jpg')));
    const [d] = await v.detector.detect(frame);
    const a = v.aligner.align(frame, d.landmarks);
    const [e1] = await v.embedder.embed([a]);
    const batch = await v.embedder.embed([a, a, a]);
    expect(Math.hypot(...e1)).toBeCloseTo(1, 5);
    for (const e of batch) for (let i = 0; i < e.length; i++) expect(e[i]).toBeCloseTo(e1[i], 4);
  });
});
