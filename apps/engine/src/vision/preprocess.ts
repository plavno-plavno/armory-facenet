// Image buffer helpers. All frames inside the engine are packed BGR24.

import type { Frame } from './types.js';

/**
 * Bilinear resize of a packed 3-channel image. Uses the same pixel-center
 * convention as cv2.resize(INTER_LINEAR): src = (dst + 0.5) * scale - 0.5.
 */
export function resizeBilinear(
  src: Uint8Array,
  sw: number,
  sh: number,
  dw: number,
  dh: number,
): Uint8Array {
  const dst = new Uint8Array(dw * dh * 3);
  const sx = sw / dw;
  const sy = sh / dh;
  const x0s = new Int32Array(dw);
  const x1s = new Int32Array(dw);
  const fxs = new Float32Array(dw);
  for (let x = 0; x < dw; x++) {
    let fx = (x + 0.5) * sx - 0.5;
    let ix = Math.floor(fx);
    fx -= ix;
    if (ix < 0) {
      ix = 0;
      fx = 0;
    }
    if (ix >= sw - 1) {
      ix = sw - 1;
      fx = 0;
    }
    x0s[x] = ix * 3;
    x1s[x] = Math.min(ix + 1, sw - 1) * 3;
    fxs[x] = fx;
  }
  for (let y = 0; y < dh; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    let iy = Math.floor(fy);
    fy -= iy;
    if (iy < 0) {
      iy = 0;
      fy = 0;
    }
    if (iy >= sh - 1) {
      iy = sh - 1;
      fy = 0;
    }
    const r0 = iy * sw * 3;
    const r1 = Math.min(iy + 1, sh - 1) * sw * 3;
    const o = y * dw * 3;
    for (let x = 0; x < dw; x++) {
      const fx = fxs[x];
      const a = x0s[x];
      const b = x1s[x];
      for (let c = 0; c < 3; c++) {
        const top = src[r0 + a + c] * (1 - fx) + src[r0 + b + c] * fx;
        const bot = src[r1 + a + c] * (1 - fx) + src[r1 + b + c] * fx;
        dst[o + x * 3 + c] = Math.round(top * (1 - fy) + bot * fy);
      }
    }
  }
  return dst;
}

/** Swap R and B in a packed 3-channel buffer (returns a new buffer). */
export function swapRB(src: Uint8Array): Uint8Array {
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    out[i] = src[i + 2];
    out[i + 1] = src[i + 1];
    out[i + 2] = src[i];
  }
  return out;
}

/** RGBA (e.g. from a canvas / VideoFrame) -> BGR24. */
export function rgbaToBgr(src: Uint8Array | Uint8ClampedArray, width: number, height: number): Uint8Array {
  const n = width * height;
  const out = new Uint8Array(n * 3);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i * 3] = src[j + 2];
    out[i * 3 + 1] = src[j + 1];
    out[i * 3 + 2] = src[j];
  }
  return out;
}

export interface DetectorInput {
  tensor: Float32Array; // [1, 3, padH, padW], BGR, no normalization
  padW: number;
  padH: number;
  scaleX: number; // resized = original * scale
  scaleY: number;
  resizedW: number;
  resizedH: number;
}

/**
 * Downscale the frame so its long side is <= longSide (never upscale), pad with
 * zeros to a multiple of 32 on the right/bottom, and lay out as planar BGR float.
 */
export function buildDetectorInput(frame: Frame, longSide: number): DetectorInput {
  const long = Math.max(frame.width, frame.height);
  const scale = long > longSide ? longSide / long : 1;
  const rw = Math.max(1, Math.round(frame.width * scale));
  const rh = Math.max(1, Math.round(frame.height * scale));
  const img = scale === 1 ? frame.data : resizeBilinear(frame.data, frame.width, frame.height, rw, rh);
  const padW = Math.ceil(rw / 32) * 32;
  const padH = Math.ceil(rh / 32) * 32;
  const plane = padW * padH;
  const tensor = new Float32Array(3 * plane);
  for (let y = 0; y < rh; y++) {
    const srcRow = y * rw * 3;
    const dstRow = y * padW;
    for (let x = 0; x < rw; x++) {
      const s = srcRow + x * 3;
      const d = dstRow + x;
      tensor[d] = img[s];
      tensor[plane + d] = img[s + 1];
      tensor[2 * plane + d] = img[s + 2];
    }
  }
  return { tensor, padW, padH, scaleX: rw / frame.width, scaleY: rh / frame.height, resizedW: rw, resizedH: rh };
}
