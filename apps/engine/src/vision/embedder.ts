// LVFace embedder (spec §6.3). Preprocessing is driven by the manifest entry.

import * as ort from 'onnxruntime-node';
import { modelKeyOf, type EmbedderManifest } from './models.js';
import type { AlignedFace, Embedder } from './types.js';

export function l2normalize(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const n = Math.sqrt(s);
  if (n > 0) for (let i = 0; i < v.length; i++) v[i] /= n;
  return v;
}

export function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Normalized mean of L2-normalized vectors (probe of a burst, spec §7.6). */
export function meanEmbedding(embs: Float32Array[]): Float32Array {
  const out = new Float32Array(embs[0].length);
  for (const e of embs) for (let i = 0; i < out.length; i++) out[i] += e[i];
  return l2normalize(out);
}

export class OnnxEmbedder implements Embedder {
  readonly modelKey: string;
  readonly dim: number;

  constructor(
    private readonly session: ort.InferenceSession,
    private readonly manifest: EmbedderManifest,
  ) {
    this.modelKey = modelKeyOf(manifest);
    this.dim = manifest.embeddingDim;
  }

  /** Build an NCHW float tensor from aligned RGB crops. */
  buildInput(faces: AlignedFace[]): Float32Array {
    const [w, h] = this.manifest.inputSize;
    const plane = w * h;
    const { mean, std, channelOrder } = this.manifest;
    const t = new Float32Array(faces.length * 3 * plane);
    faces.forEach((f, n) => {
      if (f.size !== w || w !== h) throw new Error(`Aligned face size ${f.size} does not match model input ${w}x${h}`);
      const base = n * 3 * plane;
      for (let p = 0; p < plane; p++) {
        for (let c = 0; c < 3; c++) {
          // f.rgb is RGB; pick source channel according to the model's channel order.
          const srcC = channelOrder === 'RGB' ? c : 2 - c;
          t[base + c * plane + p] = (f.rgb[p * 3 + srcC] - mean[c]) / std[c];
        }
      }
    });
    return t;
  }

  async embed(faces: AlignedFace[]): Promise<Float32Array[]> {
    if (faces.length === 0) return [];
    if (!this.manifest.dynamicBatch && faces.length > 1) {
      const out: Float32Array[] = [];
      for (const f of faces) out.push(...(await this.embed([f])));
      return out;
    }
    const [w, h] = this.manifest.inputSize;
    const input = new ort.Tensor('float32', this.buildInput(faces), [faces.length, 3, h, w]);
    const res = await this.session.run({ [this.session.inputNames[0]]: input });
    const data = res[this.session.outputNames[0]].data as Float32Array;
    const dim = data.length / faces.length;
    if (dim !== this.dim) throw new Error(`Embedding dim ${dim} != manifest ${this.dim}`);
    const out: Float32Array[] = [];
    for (let i = 0; i < faces.length; i++) out.push(l2normalize(data.slice(i * dim, (i + 1) * dim)));
    return out;
  }
}
