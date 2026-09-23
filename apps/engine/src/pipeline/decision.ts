// Match decision (spec §7.7).

import type { RecognitionStatus } from '@faceid/shared';
import type { EngineConfig } from '../config/schema.js';
import type { GalleryIndex, MatchCandidate } from '../gallery/index.js';
import { meanEmbedding } from '../vision/embedder.js';

export interface Decision {
  status: Exclude<RecognitionStatus, 'low_quality'>;
  top: MatchCandidate | null;
  score: number | null;
  secondScore: number | null;
  frameAgreement: number;
}

export function decide(s1: number | null, s2: number | null, agreement: number, m: EngineConfig['match']): Decision['status'] {
  if (s1 === null) return 'unknown';
  const second = s2 ?? -1;
  if (s1 >= m.acceptThreshold && s1 - second >= m.margin && agreement >= m.minFrameAgreement) return 'match';
  if (s1 < m.rejectThreshold) return 'unknown';
  return 'uncertain';
}

/**
 * Identify a burst: probe = normalized mean of per-frame embeddings; agreement =
 * share of frames whose own top-1 equals the probe's top-1.
 */
export function identifyBurst(index: GalleryIndex, frameEmbs: Float32Array[], m: EngineConfig['match']): Decision {
  const probe = meanEmbedding(frameEmbs);
  const [c1, c2] = index.match(probe, 2);
  if (!c1) return { status: 'unknown', top: null, score: null, secondScore: null, frameAgreement: 0 };
  let agree = 0;
  for (const e of frameEmbs) if (index.match(e, 1)[0]?.personId === c1.personId) agree++;
  const frameAgreement = agree / frameEmbs.length;
  const s2 = c2?.score ?? null;
  return { status: decide(c1.score, s2, frameAgreement, m), top: c1, score: c1.score, secondScore: s2, frameAgreement };
}
