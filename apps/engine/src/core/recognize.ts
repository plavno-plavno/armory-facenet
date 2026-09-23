// One-shot recognition on an uploaded photo (spec §10.4).

import type { EngineConfig } from '../config/schema.js';
import type { GalleryIndex } from '../gallery/index.js';
import { decide } from '../pipeline/decision.js';
import { ApiError } from '../util/errors.js';
import { publicQuality, type FaceAnalyzer } from './analyzer.js';

const r4 = (v: number) => Math.round(v * 10000) / 10000;

export class RecognizeService {
  constructor(
    private readonly analyzer: FaceAnalyzer,
    private readonly index: () => GalleryIndex,
    private readonly cfg: () => EngineConfig,
  ) {}

  async identify(buf: Buffer, topK: number) {
    const { faces } = await this.analyzer.analyzeAll(buf);
    const m = this.cfg().match;
    return {
      faces: faces.map((f) => {
        const cands = this.index().match(f.embedding, Math.max(topK, 2));
        const status = f.quality.passed ? decide(cands[0]?.score ?? null, cands[1]?.score ?? null, 1, m) : 'low_quality';
        return {
          box: f.det.box.map((v) => Math.round(v)),
          quality: { ...publicQuality(f.quality), passed: f.quality.passed, reasons: f.quality.reasons },
          status,
          candidates: cands.slice(0, topK).map((c) => ({ personId: c.personId, score: r4(c.score) })),
        };
      }),
    };
  }

  async verify(buf: Buffer, personId: string) {
    const { faces } = await this.analyzer.analyzeAll(buf);
    if (faces.length === 0) throw new ApiError('NO_FACE', 'No face detected in photo');
    // Use the largest face.
    const f = faces.reduce((a, b) => (b.det.box[2] * b.det.box[3] > a.det.box[2] * a.det.box[3] ? b : a));
    const score = this.index().scorePerson(personId, f.embedding);
    if (score === null) throw new ApiError('NOT_FOUND', 'Person has no embeddings for the current model');
    const threshold = this.cfg().match.acceptThreshold;
    return { score: r4(score), match: score >= threshold, threshold };
  }
}
