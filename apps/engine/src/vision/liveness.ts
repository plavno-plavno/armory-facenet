// Passive anti-spoofing (spec §7.9, R4): MiniFASNet ensemble from minivision-ai/Silent-Face-Anti-Spoofing.
// Reference: tests/parity/python/liveness_ref.py (CropImage._get_new_box + cv2.resize, BGR 0..255, softmax).

import * as ort from 'onnxruntime-node';
import type { LivenessManifest } from './models.js';
import { resizeBilinear } from './preprocess.js';
import type { Box, Detection, Frame, LivenessChecker, LivenessResult, LivenessSample } from './types.js';

/**
 * Crop the face box scaled by `scale` around its center (capped to fit the image), shifted - not
 * clipped - back inside the frame, resized to w x h. Box coordinates are truncated to integers like
 * the original detector output.
 */
export function livenessCrop(frame: Frame, box: Box, scale: number, w: number, h: number): Uint8Array {
  const { width: srcW, height: srcH } = frame;
  const [x, y, boxW, boxH] = box.map(Math.trunc);
  const s = Math.min((srcH - 1) / boxH, (srcW - 1) / boxW, scale);
  const newW = boxW * s;
  const newH = boxH * s;
  const cx = boxW / 2 + x;
  const cy = boxH / 2 + y;
  let x1 = cx - newW / 2;
  let y1 = cy - newH / 2;
  let x2 = cx + newW / 2;
  let y2 = cy + newH / 2;
  if (x1 < 0) {
    x2 -= x1;
    x1 = 0;
  }
  if (y1 < 0) {
    y2 -= y1;
    y1 = 0;
  }
  if (x2 > srcW - 1) {
    x1 -= x2 - srcW + 1;
    x2 = srcW - 1;
  }
  if (y2 > srcH - 1) {
    y1 -= y2 - srcH + 1;
    y2 = srcH - 1;
  }
  const [l, t, r, b] = [x1, y1, x2, y2].map(Math.trunc);
  const cw = r - l + 1;
  const ch = b - t + 1;
  const crop = new Uint8Array(cw * ch * 3);
  for (let row = 0; row < ch; row++) {
    const src = ((t + row) * srcW + l) * 3;
    crop.set(frame.data.subarray(src, src + cw * 3), row * cw * 3);
  }
  return resizeBilinear(crop, cw, ch, w, h);
}

export interface LivenessModel {
  session: ort.InferenceSession;
  manifest: LivenessManifest;
}

export class OnnxLivenessChecker implements LivenessChecker {
  constructor(
    private readonly models: LivenessModel[],
    private readonly cfg: () => { enabled: boolean; threshold: number },
  ) {}

  enabled(): boolean {
    return this.cfg().enabled && this.models.length > 0;
  }

  sample(frame: Frame, det: Detection): LivenessSample {
    return {
      crops: this.models.map(({ manifest: m }) => livenessCrop(frame, det.box, m.cropScale, m.inputSize[1], m.inputSize[0])),
    };
  }

  /** Real-class probability per sample and model: result[sample][model]. */
  async probabilities(samples: LivenessSample[]): Promise<number[][]> {
    const out = samples.map(() => [] as number[]);
    for (const [k, { session, manifest: m }] of this.models.entries()) {
      const [h, w] = m.inputSize;
      const plane = h * w;
      const input = new Float32Array(samples.length * 3 * plane);
      samples.forEach((s, n) => {
        const crop = s.crops[k];
        const base = n * 3 * plane;
        for (let i = 0; i < plane; i++) {
          input[base + i] = crop[i * 3];
          input[base + plane + i] = crop[i * 3 + 1];
          input[base + 2 * plane + i] = crop[i * 3 + 2];
        }
      });
      const res = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', input, [samples.length, 3, h, w]) });
      const logits = res[session.outputNames[0]].data as Float32Array;
      const classes = logits.length / samples.length;
      for (let n = 0; n < samples.length; n++) {
        const row = logits.subarray(n * classes, (n + 1) * classes);
        const max = Math.max(...row);
        let sum = 0;
        for (const v of row) sum += Math.exp(v - max);
        out[n].push(Math.exp(row[m.realClass] - max) / sum);
      }
    }
    return out;
  }

  async check(samples: LivenessSample[]): Promise<LivenessResult> {
    const probs = await this.probabilities(samples);
    const perFrame = probs.map((p) => p.reduce((a, v) => a + v, 0) / p.length);
    const score = perFrame.reduce((a, v) => a + v, 0) / perFrame.length;
    return { live: score >= this.cfg().threshold, score };
  }
}
