// YuNet face detector: output decoding + NMS (spec §6.1). Mirrors
// cv::FaceDetectorYN::postProcess from OpenCV modules/objdetect/src/face_detect.cpp.

import * as ort from 'onnxruntime-node';
import { buildDetectorInput } from './preprocess.js';
import type { Box, Detection, Detector, Frame, Landmarks5 } from './types.js';

export const YUNET_STRIDES = [8, 16, 32] as const;

export interface YuNetOptions {
  inputLongSide: number;
  scoreThreshold: number;
  nmsThreshold: number;
  topK?: number;
}

export interface RawOutputs {
  cls: Float32Array[];
  obj: Float32Array[];
  bbox: Float32Array[];
  kps: Float32Array[];
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Decode raw YuNet outputs at the padded input resolution. */
export function decodeYuNet(out: RawOutputs, padW: number, padH: number, scoreThreshold: number): Detection[] {
  const faces: Detection[] = [];
  YUNET_STRIDES.forEach((s, si) => {
    const cols = Math.floor(padW / s);
    const rows = Math.floor(padH / s);
    const cls = out.cls[si];
    const obj = out.obj[si];
    const bbox = out.bbox[si];
    const kps = out.kps[si];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const i = r * cols + c;
        // Math.fround keeps float32 semantics of the reference implementation.
        const score = Math.fround(Math.sqrt(Math.fround(clamp01(cls[i]) * clamp01(obj[i]))));
        if (score < scoreThreshold) continue;
        const cx = (c + bbox[i * 4]) * s;
        const cy = (r + bbox[i * 4 + 1]) * s;
        const w = Math.exp(bbox[i * 4 + 2]) * s;
        const h = Math.exp(bbox[i * 4 + 3]) * s;
        const lm: [number, number][] = [];
        for (let k = 0; k < 5; k++) {
          lm.push([(kps[i * 10 + 2 * k] + c) * s, (kps[i * 10 + 2 * k + 1] + r) * s]);
        }
        faces.push({ box: [cx - w / 2, cy - h / 2, w, h], landmarks: lm as Landmarks5, score });
      }
    }
  });
  return faces;
}

/** IoU of two integer rects, same as cv::Rect area semantics (no +1). */
export function rectIoU(a: Box, b: Box): number {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const uni = a[2] * a[3] + b[2] * b[3] - inter;
  return uni <= 0 ? 0 : inter / uni;
}

/**
 * Greedy NMS equivalent to cv::dnn::NMSBoxes on Rect2i: boxes are truncated to
 * int, candidates sorted by score (stable, descending), eta = 1.
 */
export function nms(dets: Detection[], scoreThreshold: number, iouThreshold: number, topK = 5000): Detection[] {
  const cand = dets
    .map((d, idx) => ({ d, idx, rect: d.box.map((v) => Math.trunc(v)) as Box }))
    .filter((c) => c.d.score > scoreThreshold) // NMSBoxes keeps strictly greater
    .sort((a, b) => b.d.score - a.d.score || a.idx - b.idx)
    .slice(0, topK);
  const kept: typeof cand = [];
  for (const c of cand) {
    let keep = true;
    for (const k of kept) {
      if (rectIoU(c.rect, k.rect) > iouThreshold) {
        keep = false;
        break;
      }
    }
    if (keep) kept.push(c);
  }
  return kept.map((k) => k.d);
}

export class YuNetDetector implements Detector {
  private readonly outNames: string[];

  constructor(
    private readonly session: ort.InferenceSession,
    private opts: YuNetOptions,
  ) {
    const names: string[] = [];
    for (const kind of ['cls', 'obj', 'bbox', 'kps']) for (const s of YUNET_STRIDES) names.push(`${kind}_${s}`);
    this.outNames = names;
  }

  /** Thresholds and input size are applied live (config PATCH). */
  setOptions(opts: YuNetOptions): void {
    this.opts = { ...opts };
  }

  async detect(frame: Frame, longSideOverride?: number): Promise<Detection[]> {
    const inp = buildDetectorInput(frame, longSideOverride ?? this.opts.inputLongSide);
    const tensor = new ort.Tensor('float32', inp.tensor, [1, 3, inp.padH, inp.padW]);
    const res = await this.session.run({ [this.session.inputNames[0]]: tensor }, this.outNames);
    const get = (k: string) => res[k].data as Float32Array;
    const raw: RawOutputs = {
      cls: YUNET_STRIDES.map((s) => get(`cls_${s}`)),
      obj: YUNET_STRIDES.map((s) => get(`obj_${s}`)),
      bbox: YUNET_STRIDES.map((s) => get(`bbox_${s}`)),
      kps: YUNET_STRIDES.map((s) => get(`kps_${s}`)),
    };
    const decoded = decodeYuNet(raw, inp.padW, inp.padH, this.opts.scoreThreshold);
    const kept = nms(decoded, this.opts.scoreThreshold, this.opts.nmsThreshold, this.opts.topK ?? 5000);
    const ix = 1 / inp.scaleX;
    const iy = 1 / inp.scaleY;
    return kept.map((d) => ({
      score: d.score,
      box: [d.box[0] * ix, d.box[1] * iy, d.box[2] * ix, d.box[3] * iy],
      landmarks: d.landmarks.map(([x, y]) => [x * ix, y * iy]) as Landmarks5,
    }));
  }
}
