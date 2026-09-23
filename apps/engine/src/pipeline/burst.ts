// Burst of frames of one track, best-frame selection (spec §7.5).

import type { EngineConfig } from '../config/schema.js';
import type { AlignedFace, Detection, Frame, LivenessChecker, LivenessResult, LivenessSample, QualityReport } from '../vision/types.js';

export interface BurstCandidate {
  det: Detection;
  aligned: AlignedFace;
  quality: QualityReport;
  frame?: Frame; // kept only when needed (snapshot / capture)
  liveness?: LivenessSample; // good frames only, while liveness is enabled
}

/** Liveness of the frames about to be identified; null when the check is off or has no samples. */
export async function checkLiveness(checker: LivenessChecker, best: BurstCandidate[]): Promise<LivenessResult | null> {
  if (!checker.enabled()) return null;
  const samples = best.flatMap((c) => (c.liveness ? [c.liveness] : []));
  return samples.length ? checker.check(samples) : null;
}

export class Burst {
  readonly startTs: number;
  considered = 0;
  readonly good: BurstCandidate[] = [];
  readonly reasons: Record<string, number> = {};

  constructor(startTs: number) {
    this.startTs = startTs;
  }

  add(c: BurstCandidate): void {
    this.considered++;
    if (c.quality.passed) this.good.push(c);
    else for (const r of c.quality.reasons) this.reasons[r] = (this.reasons[r] ?? 0) + 1;
  }

  /** Complete after maxFrames frames or windowMs since the first frame. */
  isComplete(now: number, cfg: EngineConfig['burst']): boolean {
    return this.considered >= cfg.maxFrames || now - this.startTs >= cfg.windowMs;
  }

  enoughGood(cfg: EngineConfig['burst']): boolean {
    return this.good.length >= cfg.minGood;
  }

  best(k: number): BurstCandidate[] {
    return [...this.good].sort((a, b) => b.quality.qualityScore - a.quality.qualityScore).slice(0, k);
  }
}
