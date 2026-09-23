// Score distributions and threshold suggestions (spec §10.6 gallery-scores, §15.3).

import type { EngineConfig } from '../config/schema.js';
import { dot } from '../vision/embedder.js';
import type { GalleryIndex } from './index.js';

export interface Histogram {
  min: number;
  max: number;
  counts: number[];
}

export function histogram(values: number[], bins: number, min = -0.2, max = 1): Histogram {
  const counts = new Array(bins).fill(0);
  const w = (max - min) / bins;
  for (const v of values) counts[Math.min(bins - 1, Math.max(0, Math.floor((v - min) / w)))]++;
  return { min, max, counts };
}

export function stats(v: number[]) {
  if (v.length === 0) return { n: 0, mean: null, std: null, min: null, max: null };
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / v.length);
  return { n: v.length, mean, std, min: Math.min(...v), max: Math.max(...v) };
}

/** Threshold such that the share of impostor scores >= t is <= far. */
export function thresholdAtFar(impostor: number[], far: number): number | null {
  if (impostor.length === 0) return null;
  const s = [...impostor].sort((a, b) => b - a);
  const k = Math.floor(far * s.length);
  if (k >= s.length) return s[s.length - 1];
  return s[k] + 1e-6; // strictly above the (k+1)-th highest impostor score
}

export function rateAtThreshold(scores: number[], t: number, above: boolean): number {
  if (scores.length === 0) return 0;
  return scores.filter((s) => (above ? s >= t : s < t)).length / scores.length;
}

export function eer(genuine: number[], impostor: number[]): { threshold: number; rate: number } | null {
  if (!genuine.length || !impostor.length) return null;
  let best = { threshold: 0, rate: 1, gap: Infinity };
  for (let t = -0.2; t <= 1; t += 0.001) {
    const far = rateAtThreshold(impostor, t, true);
    const frr = rateAtThreshold(genuine, t, false);
    const gap = Math.abs(far - frr);
    if (gap < best.gap) best = { threshold: t, rate: (far + frr) / 2, gap };
  }
  return { threshold: Math.round(best.threshold * 1000) / 1000, rate: best.rate };
}

/** Deterministic PRNG (mulberry32) so repeated calls sample the same pairs. */
function rng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function galleryScoreDistributions(index: GalleryIndex, bins: number, maxImpostorPairs: number, match: EngineConfig['match']) {
  const persons = index.personIds();
  const embs = persons.map((p) => index.personEmbeddings(p));
  const genuine: number[] = [];
  for (const list of embs) for (let i = 0; i < list.length; i++) for (let j = i + 1; j < list.length; j++) genuine.push(dot(list[i], list[j]));
  const impostor: number[] = [];
  const flat = embs.flatMap((list, pi) => list.map((e) => ({ pi, e })));
  const totalPairs = (flat.length * (flat.length - 1)) / 2;
  if (totalPairs <= maxImpostorPairs) {
    for (let i = 0; i < flat.length; i++) for (let j = i + 1; j < flat.length; j++) if (flat[i].pi !== flat[j].pi) impostor.push(dot(flat[i].e, flat[j].e));
  } else {
    const r = rng(42);
    for (let k = 0; k < maxImpostorPairs; k++) {
      const a = flat[Math.floor(r() * flat.length)];
      const b = flat[Math.floor(r() * flat.length)];
      if (a.pi !== b.pi) impostor.push(dot(a.e, b.e));
    }
  }
  const round = (v: number | null) => (v === null ? null : Math.round(v * 10000) / 10000);
  return {
    note: 'Photo-vs-photo scores inside the gallery. Final thresholds must come from camera calibration (npm run calibrate).',
    genuine: { ...stats(genuine), histogram: histogram(genuine, bins) },
    impostor: { ...stats(impostor), histogram: histogram(impostor, bins) },
    current: {
      acceptThreshold: match.acceptThreshold,
      far: round(rateAtThreshold(impostor, match.acceptThreshold, true)),
      frr: round(rateAtThreshold(genuine, match.acceptThreshold, false)),
    },
    suggested: {
      acceptAtFar1e3: round(thresholdAtFar(impostor, 1e-3)),
      acceptAtFar1e4: round(thresholdAtFar(impostor, 1e-4)),
      eer: eer(genuine, impostor),
    },
  };
}
