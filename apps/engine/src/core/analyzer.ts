// Single-image face analysis used by enrollment, identify/verify and reindexing (spec §9.1).

import type { EngineConfig } from '../config/schema.js';
import { enrollQuality } from '../config/schema.js';
import { ApiError } from '../util/errors.js';
import { decodeImage, encodePngRgb, UnsupportedImageError } from '../vision/image.js';
import type { VisionStack } from '../vision/index.js';
import { DefaultQualityAssessor, type QualityThresholds } from '../vision/quality.js';
import type { AlignedFace, Box, Detection, Frame, QualityReport } from '../vision/types.js';
import { rectIoU } from '../vision/yunet.js';

export interface AnalyzedFace {
  det: Detection;
  aligned: AlignedFace;
  quality: QualityReport;
  embedding: Float32Array;
}

export interface AnalyzedPhoto extends AnalyzedFace {
  frame: Frame;
  originalJpeg: Buffer;
  alignedPng: Buffer;
}

const roundBox = (b: Box): Box => b.map((v) => Math.round(v)) as Box;

export function publicQuality(q: QualityReport) {
  const r = (v: number, d = 3) => Math.round(v * 10 ** d) / 10 ** d;
  return { detScore: r(q.detScore), faceSize: r(q.faceSize, 1), yaw: r(q.yaw), rollDeg: r(q.rollDeg, 1), sharpness: r(q.sharpness, 1), brightness: r(q.brightness, 1) };
}

export class FaceAnalyzer {
  constructor(
    private readonly vision: VisionStack,
    private readonly cfg: () => EngineConfig,
  ) {}

  /** Detect faces; if nothing is found, retry at half resolution for faces that are too large for YuNet. */
  async detectAll(frame: Frame): Promise<Detection[]> {
    const long = this.cfg().detector.inputLongSide;
    let dets = await this.vision.detector.detect(frame, long);
    if (dets.length === 0 && Math.max(frame.width, frame.height) > long / 2) {
      dets = await this.vision.detector.detect(frame, Math.round(long / 2));
    }
    return dets;
  }

  async decode(buf: Buffer) {
    const c = this.cfg().enroll;
    if (buf.length > c.maxFileMb * 1024 * 1024) {
      throw new ApiError('FILE_TOO_LARGE', `File exceeds ${c.maxFileMb} MB`, { maxFileMb: c.maxFileMb });
    }
    try {
      return await decodeImage(buf);
    } catch (e) {
      if (e instanceof UnsupportedImageError) throw new ApiError('UNSUPPORTED_FORMAT', 'Supported formats: JPEG, PNG, WebP');
      throw e;
    }
  }

  /** Choose exactly one face (or the one matching `faceBox`). */
  pickFace(dets: Detection[], faceBox?: Box): Detection {
    if (dets.length === 0) throw new ApiError('NO_FACE', 'No face detected in photo');
    if (faceBox) {
      let best: Detection | undefined;
      let bestIoU = 0;
      for (const d of dets) {
        const v = rectIoU(d.box, faceBox);
        if (v > bestIoU) {
          bestIoU = v;
          best = d;
        }
      }
      if (!best || bestIoU < 0.3) {
        throw new ApiError('NO_FACE', 'No detected face matches faceBox', { boxes: dets.map((d) => roundBox(d.box)) });
      }
      return best;
    }
    if (dets.length > 1) {
      throw new ApiError('MULTIPLE_FACES', `${dets.length} faces detected; pass faceBox to choose one`, {
        boxes: dets.map((d) => roundBox(d.box)),
      });
    }
    return dets[0];
  }

  assess(frame: Frame, det: Detection, aligned: AlignedFace, thresholds: QualityThresholds): QualityReport {
    return new DefaultQualityAssessor(thresholds).assess(frame, det, aligned);
  }

  /** Full enrollment analysis of an uploaded photo; throws ApiError on any rule violation. */
  async analyzeForEnrollment(buf: Buffer, faceBox?: Box): Promise<AnalyzedPhoto> {
    const cfg = this.cfg();
    const img = await this.decode(buf);
    const { frame } = img;
    if (Math.min(frame.width, frame.height) < cfg.enroll.minShortSide) {
      throw new ApiError('IMAGE_TOO_SMALL', `Short side must be >= ${cfg.enroll.minShortSide}px`, {
        width: frame.width,
        height: frame.height,
      });
    }
    const det = this.pickFace(await this.detectAll(frame), faceBox);
    return this.finishEnrollment(frame, det, img.jpeg);
  }

  /** Quality gate + embedding for an already chosen face (upload or camera capture). */
  async finishEnrollment(frame: Frame, det: Detection, originalJpeg: Buffer): Promise<AnalyzedPhoto> {
    const aligned = this.vision.aligner.align(frame, det.landmarks);
    const quality = this.assess(frame, det, aligned, enrollQuality(this.cfg()));
    if (!quality.passed) {
      throw new ApiError('LOW_QUALITY', `Photo quality too low: ${quality.reasons.join(', ')}`, {
        reasons: quality.reasons,
        quality: publicQuality(quality),
      });
    }
    const [embedding] = await this.vision.embedder.embed([aligned]);
    return { frame, det, aligned, quality, embedding, originalJpeg, alignedPng: await encodePngRgb(aligned.rgb, 112, 112) };
  }

  /** All faces of an image with stream-level quality (identify endpoint). */
  async analyzeAll(buf: Buffer): Promise<{ frame: Frame; faces: AnalyzedFace[] }> {
    const { frame } = await this.decode(buf);
    const dets = await this.detectAll(frame);
    const aligned = dets.map((d) => this.vision.aligner.align(frame, d.landmarks));
    const embs = await this.vision.embedder.embed(aligned);
    const q = this.cfg().quality;
    return {
      frame,
      faces: dets.map((det, i) => ({ det, aligned: aligned[i], embedding: embs[i], quality: this.assess(frame, det, aligned[i], q) })),
    };
  }
}
