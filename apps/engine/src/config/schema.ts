// Engine configuration (spec §11). Defaults live in the schema.

import { z } from 'zod';

const Brightness = z.tuple([z.number(), z.number()]);

/**
 * Enrollment thresholds (stricter than stream). Keys without a default inherit from `quality`.
 */
const EnrollQuality = z.object({
  minFaceSize: z.number().min(0).default(120),
  maxYaw: z.number().min(0).default(0.15),
  maxRollDeg: z.number().min(0).max(90).default(15),
  minSharpness: z.number().min(0).default(150),
  minDetScore: z.number().min(0).max(1).optional(),
  minInterocular: z.number().min(0).optional(),
  brightness: Brightness.optional(),
  requireInFrame: z.boolean().optional(),
});

export const QualityConfig = z.object({
  minDetScore: z.number().min(0).max(1).default(0.9),
  minFaceSize: z.number().min(0).default(80),
  minInterocular: z.number().min(0).default(30),
  maxYaw: z.number().min(0).default(0.25),
  maxRollDeg: z.number().min(0).max(90).default(25),
  minSharpness: z.number().min(0).default(100),
  brightness: Brightness.default([40, 220]),
  requireInFrame: z.boolean().default(true),
  weights: z
    .object({
      detScore: z.number().min(0).default(1),
      faceSize: z.number().min(0).default(1),
      yaw: z.number().min(0).default(1),
      sharpness: z.number().min(0).default(1),
    })
    .prefault({}),
});

const Webhook = z.object({
  url: z.url(),
  events: z.array(z.string()).default(['recognition.result']),
  secret: z.string().min(8),
  statuses: z.array(z.enum(['match', 'unknown', 'uncertain', 'low_quality', 'spoof'])).optional(),
});

export const EngineConfig = z.object({
  server: z
    .object({
      host: z.string().default('127.0.0.1'),
      port: z.number().int().min(0).max(65535).default(47810), // 0 = any free port
      tls: z.object({ cert: z.string(), key: z.string() }).nullable().default(null),
      enrollRatePerSec: z.number().int().min(1).default(10), // spec §12.3
      corsOrigins: z.array(z.string()).default([]), // exact origins, e.g. the Electron admin UI "faceid://admin"

    })
    .prefault({}),
  dataDir: z.string().nullable().default(null),
  models: z
    .object({
      embedder: z.string().default('lvface-s-glint360k'),
      executionProvider: z.enum(['auto', 'cpu', 'cuda', 'dml']).default('auto'),
      intraOpThreads: z.number().int().min(0).default(0),
    })
    .prefault({}),
  detector: z
    .object({
      inputLongSide: z.number().int().min(160).max(1920).default(640),
      scoreThreshold: z.number().min(0).max(1).default(0.9),
      nmsThreshold: z.number().min(0).max(1).default(0.3),
    })
    .prefault({}),
  pipeline: z
    .object({
      detectFps: z.number().positive().max(60).default(10),
      maxConcurrentTracks: z.number().int().min(1).default(3),
      maxRetries: z.number().int().min(0).default(2),
      cooldownMs: z.number().int().min(0).default(10000),
    })
    .prefault({}),
  tracker: z
    .object({
      iouThreshold: z.number().min(0).max(1).default(0.3),
      ttlMs: z.number().int().min(0).default(1000),
    })
    .prefault({}),
  burst: z
    .object({
      maxFrames: z.number().int().min(1).default(10),
      windowMs: z.number().int().min(0).default(1000),
      topK: z.number().int().min(1).default(5),
      minGood: z.number().int().min(1).default(3),
    })
    .prefault({}),
  quality: QualityConfig.prefault({}),
  match: z
    .object({
      acceptThreshold: z.number().min(-1).max(1).default(0.45),
      rejectThreshold: z.number().min(-1).max(1).default(0.3),
      margin: z.number().min(0).max(2).default(0.05),
      minFrameAgreement: z.number().min(0).max(1).default(0.6),
    })
    .prefault({}),
  // Anti-spoofing (spec §7.9, R4). The threshold is on the mean real-face probability of the burst;
  // like the match thresholds it must be calibrated on the target camera.
  liveness: z
    .object({
      enabled: z.boolean().default(true),
      threshold: z.number().min(0).max(1).default(0.5),
    })
    .prefault({}),
  enroll: z
    .object({
      maxPhotos: z.number().int().min(1).default(10),
      maxFileMb: z.number().positive().default(10),
      minShortSide: z.number().int().min(1).default(300),
      captureFrames: z.number().int().min(1).default(3),
      captureTimeoutMs: z.number().int().min(1000).default(10000),
      quality: EnrollQuality.prefault({}),
    })
    .prefault({}),
  events: z
    .object({
      storeSnapshots: z.boolean().default(false),
      snapshotRetentionDays: z.number().int().min(1).default(7),
      logRetentionDays: z.number().int().min(1).default(90),
    })
    .prefault({}),
  privacy: z
    .object({
      requireConsent: z.boolean().default(true),
      encryptAtRest: z.boolean().default(true),
    })
    .prefault({}),
  webhooks: z.array(Webhook).default([]),
  log: z.object({ level: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace']).default('info') }).prefault({}),
  cameraProfile: z.string().nullable().default(null),
});

export type EngineConfig = z.infer<typeof EngineConfig>;
export type QualityConfig = z.infer<typeof QualityConfig>;

/** Config paths whose change only takes effect after an engine restart. */
export const RESTART_REQUIRED = ['server', 'dataDir', 'models', 'privacy.encryptAtRest'];

export function defaultConfig(): EngineConfig {
  return EngineConfig.parse({});
}

/** Stream thresholds merged with the stricter enrollment overrides. */
export function enrollQuality(cfg: EngineConfig): QualityConfig {
  const o = Object.fromEntries(Object.entries(cfg.enroll.quality).filter(([, v]) => v !== undefined));
  return { ...cfg.quality, ...o, weights: cfg.quality.weights };
}
