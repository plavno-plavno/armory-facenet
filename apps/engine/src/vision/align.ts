// Face alignment to the ArcFace 112x112 template (spec §6.2). Reference:
// insightface.utils.face_align.norm_crop (skimage SimilarityTransform + cv2.warpAffine).

import type { AlignedFace, Aligner, Frame, Landmarks5, Point } from './types.js';

export const ARCFACE_TEMPLATE: Landmarks5 = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

export const ALIGNED_SIZE = 112;

/**
 * Least-squares similarity transform src -> dst without reflection (Umeyama).
 * In 2D the proper similarity [[a,-b],[b,a]] has a closed-form solution that
 * equals Umeyama's result. Returns a 2x3 row-major matrix.
 */
export function estimateSimilarity(src: Point[], dst: Point[]): number[] {
  const n = src.length;
  let smx = 0;
  let smy = 0;
  let dmx = 0;
  let dmy = 0;
  for (let i = 0; i < n; i++) {
    smx += src[i][0];
    smy += src[i][1];
    dmx += dst[i][0];
    dmy += dst[i][1];
  }
  smx /= n;
  smy /= n;
  dmx /= n;
  dmy /= n;
  let num1 = 0;
  let num2 = 0;
  let den = 0;
  for (let i = 0; i < n; i++) {
    const sx = src[i][0] - smx;
    const sy = src[i][1] - smy;
    const dx = dst[i][0] - dmx;
    const dy = dst[i][1] - dmy;
    num1 += sx * dx + sy * dy;
    num2 += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }
  if (den === 0) throw new Error('Degenerate landmarks');
  const a = num1 / den;
  const b = num2 / den;
  const tx = dmx - (a * smx - b * smy);
  const ty = dmy - (b * smx + a * smy);
  return [a, -b, tx, b, a, ty];
}

export function invertAffine(m: number[]): number[] {
  const [a, b, c, d, e, f] = m;
  const det = a * e - b * d;
  if (det === 0) throw new Error('Singular affine transform');
  const ia = e / det;
  const ib = -b / det;
  const id = -d / det;
  const ie = a / det;
  return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}

const INTER_TAB = 32; // cv2 quantizes bilinear weights to 1/32 px

/**
 * Warp a packed 3-channel image with an affine matrix (src -> dst), bilinear,
 * constant 0 border. Output keeps the channel order of the input.
 */
export function warpAffine(
  src: Uint8Array,
  sw: number,
  sh: number,
  m: number[],
  dw: number,
  dh: number,
): Uint8Array {
  const inv = invertAffine(m);
  const out = new Uint8Array(dw * dh * 3);
  const px = (x: number, y: number, c: number): number =>
    x >= 0 && y >= 0 && x < sw && y < sh ? src[(y * sw + x) * 3 + c] : 0;
  for (let y = 0; y < dh; y++) {
    for (let x = 0; x < dw; x++) {
      const fx = Math.round((inv[0] * x + inv[1] * y + inv[2]) * INTER_TAB) / INTER_TAB;
      const fy = Math.round((inv[3] * x + inv[4] * y + inv[5]) * INTER_TAB) / INTER_TAB;
      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const ax = fx - x0;
      const ay = fy - y0;
      const o = (y * dw + x) * 3;
      for (let c = 0; c < 3; c++) {
        const v =
          (px(x0, y0, c) * (1 - ax) + px(x0 + 1, y0, c) * ax) * (1 - ay) +
          (px(x0, y0 + 1, c) * (1 - ax) + px(x0 + 1, y0 + 1, c) * ax) * ay;
        out[o + c] = Math.round(v);
      }
    }
  }
  return out;
}

export class ArcFaceAligner implements Aligner {
  align(frame: Frame, lm: Landmarks5): AlignedFace {
    const m = estimateSimilarity(lm, ARCFACE_TEMPLATE);
    const bgr = warpAffine(frame.data, frame.width, frame.height, m, ALIGNED_SIZE, ALIGNED_SIZE);
    const rgb = new Uint8Array(bgr.length);
    for (let i = 0; i < bgr.length; i += 3) {
      rgb[i] = bgr[i + 2];
      rgb[i + 1] = bgr[i + 1];
      rgb[i + 2] = bgr[i];
    }
    return { rgb, size: 112, transform: m };
  }
}
