// Face quality metrics (spec §7.5).

import type { AlignedFace, Detection, Frame, QualityAssessor, QualityReport } from './types.js';

export interface QualityThresholds {
  minDetScore: number;
  minFaceSize: number;
  minInterocular: number;
  maxYaw: number;
  maxRollDeg: number;
  minSharpness: number;
  brightness: [number, number];
  requireInFrame: boolean;
  weights: { detScore: number; faceSize: number; yaw: number; sharpness: number };
}

/** Grayscale (luma) of an RGB crop. */
export function toGray(rgb: Uint8Array): Float32Array {
  const g = new Float32Array(rgb.length / 3);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2];
  return g;
}

/** Variance of the 4-neighbour Laplacian over interior pixels. */
export function laplacianVariance(gray: Float32Array, w: number, h: number): number {
  let sum = 0;
  let sum2 = 0;
  let n = 0;
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const v = gray[i - w] + gray[i + w] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      sum += v;
      sum2 += v * v;
      n++;
    }
  }
  const mean = sum / n;
  return sum2 / n - mean * mean;
}

export function meanOf(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s / a.length;
}

/** Geometry-only metrics derived from detector output. */
export function geometryMetrics(det: Detection, frameW: number, frameH: number) {
  const [re, le, nose] = det.landmarks;
  const interocular = Math.hypot(le[0] - re[0], le[1] - re[1]);
  const eyesMidX = (re[0] + le[0]) / 2;
  const yaw = interocular > 0 ? Math.abs(nose[0] - eyesMidX) / interocular : Infinity;
  const rollDeg = Math.abs((Math.atan2(le[1] - re[1], le[0] - re[0]) * 180) / Math.PI);
  const [x, y, w, h] = det.box;
  const faceSize = Math.min(w, h);
  const inFrame = x >= 0 && y >= 0 && x + w <= frameW && y + h <= frameH;
  return { interocular, yaw, rollDeg, faceSize, inFrame };
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

export class DefaultQualityAssessor implements QualityAssessor {
  constructor(private readonly t: QualityThresholds) {}

  assess(frame: Frame, det: Detection, face: AlignedFace): QualityReport {
    const t = this.t;
    const g = geometryMetrics(det, frame.width, frame.height);
    const gray = toGray(face.rgb);
    const sharpness = laplacianVariance(gray, face.size, face.size);
    const brightness = meanOf(gray);
    const reasons: string[] = [];
    if (det.score < t.minDetScore) reasons.push('detScore');
    if (g.faceSize < t.minFaceSize) reasons.push('faceSize');
    if (g.interocular < t.minInterocular) reasons.push('interocular');
    if (g.yaw > t.maxYaw) reasons.push('yaw');
    if (g.rollDeg > t.maxRollDeg) reasons.push('roll');
    if (sharpness < t.minSharpness) reasons.push('sharpness');
    if (brightness < t.brightness[0] || brightness > t.brightness[1]) reasons.push('brightness');
    if (t.requireInFrame && !g.inFrame) reasons.push('inFrame');

    const w = t.weights;
    const wsum = w.detScore + w.faceSize + w.yaw + w.sharpness || 1;
    const qualityScore =
      (w.detScore * clamp01(det.score) +
        w.faceSize * clamp01(g.faceSize / (2 * t.minFaceSize)) +
        w.yaw * clamp01(1 - g.yaw / Math.max(t.maxYaw, 1e-6)) +
        w.sharpness * clamp01(sharpness / (4 * Math.max(t.minSharpness, 1)))) /
      wsum;

    return {
      detScore: det.score,
      faceSize: g.faceSize,
      interocular: g.interocular,
      yaw: g.yaw,
      rollDeg: g.rollDeg,
      sharpness,
      brightness,
      inFrame: g.inFrame,
      passed: reasons.length === 0,
      reasons,
      qualityScore,
    };
  }
}
