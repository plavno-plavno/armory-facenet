// Core vision interfaces (spec §7.1).

export interface Frame {
  data: Uint8Array; // BGR24, row-major
  width: number;
  height: number;
  ts: number; // ms, monotonic
  sourceId: string;
}

export type Point = [number, number];
export type Landmarks5 = [Point, Point, Point, Point, Point];
export type Box = [number, number, number, number]; // x, y, w, h

export interface Detection {
  box: Box;
  landmarks: Landmarks5;
  score: number;
}

export interface AlignedFace {
  rgb: Uint8Array; // 112*112*3
  size: 112;
  transform: number[]; // 2x3 row-major, source -> aligned
}

export interface Detector {
  detect(frame: Frame): Promise<Detection[]>;
}

export interface Aligner {
  align(frame: Frame, lm: Landmarks5): AlignedFace;
}

export interface Embedder {
  readonly modelKey: string;
  readonly dim: number;
  embed(faces: AlignedFace[]): Promise<Float32Array[]>; // L2-normalized
}

export interface QualityReport {
  detScore: number;
  faceSize: number;
  interocular: number;
  yaw: number;
  rollDeg: number;
  sharpness: number;
  brightness: number;
  inFrame: boolean;
  passed: boolean;
  reasons: string[];
  qualityScore: number;
}

export interface QualityAssessor {
  assess(frame: Frame, det: Detection, face: AlignedFace): QualityReport;
}

export interface LivenessChecker {
  check(faces: AlignedFace[], frames: Frame[]): Promise<{ live: boolean; score: number }>;
}

export class NoopLivenessChecker implements LivenessChecker {
  async check(): Promise<{ live: boolean; score: number }> {
    return { live: true, score: 1 };
  }
}
