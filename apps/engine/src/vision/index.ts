// Loads models from the manifest and wires detector/aligner/embedder together.

import { existsSync } from 'node:fs';
import path from 'node:path';
import { ArcFaceAligner } from './align.js';
import { OnnxEmbedder } from './embedder.js';
import type { LivenessModel } from './liveness.js';
import {
  createSession,
  loadManifest,
  selectEmbedder,
  verifyModelFile,
  type EmbedderManifest,
  type ExecutionProviderPref,
  type ModelManifest,
} from './models.js';
import { YuNetDetector, type YuNetOptions } from './yunet.js';

export interface VisionOptions {
  modelsDir: string;
  embedderId?: string;
  executionProvider: ExecutionProviderPref;
  intraOpThreads?: number;
  detector: YuNetOptions;
  verifyChecksums?: boolean;
}

export interface VisionStack {
  manifest: ModelManifest;
  embedderManifest: EmbedderManifest;
  detector: YuNetDetector;
  aligner: ArcFaceAligner;
  embedder: OnnxEmbedder;
  /** Anti-spoofing models; empty when the manifest lists none or their files are missing. */
  liveness: LivenessModel[];
  info: {
    detector: { id: string; executionProvider: string };
    embedder: { modelKey: string; executionProvider: string; fallbackReason: string | null };
    liveness: { models: string[]; missing: string[] };
  };
}

export async function loadVision(opts: VisionOptions): Promise<VisionStack> {
  const manifest = loadManifest(opts.modelsDir);
  const embM = selectEmbedder(manifest, opts.embedderId);
  const verify = opts.verifyChecksums ?? true;
  const pathOf = async (e: { file: string; sha256: string }) =>
    verify ? verifyModelFile(opts.modelsDir, e) : `${opts.modelsDir}/${e.file}`;
  const [detFile, embFile] = await Promise.all([pathOf(manifest.detector), pathOf(embM)]);
  // The detector is small; CPU avoids GPU transfer overhead for its variable input shapes.
  const det = await createSession(detFile, 'cpu', opts.intraOpThreads);
  const emb = await createSession(embFile, opts.executionProvider, opts.intraOpThreads);
  const embedder = new OnnxEmbedder(emb.session, embM);
  // Liveness is optional: installs predating it have no files, the engine still starts (with a warning).
  const livM = manifest.liveness ?? [];
  const missing = livM.filter((m) => !existsSync(path.join(opts.modelsDir, m.file))).map((m) => m.file);
  const liveness: LivenessModel[] = missing.length
    ? []
    : await Promise.all(livM.map(async (m) => ({ manifest: m, session: (await createSession(await pathOf(m), 'cpu', opts.intraOpThreads)).session })));
  return {
    manifest,
    embedderManifest: embM,
    detector: new YuNetDetector(det.session, opts.detector),
    aligner: new ArcFaceAligner(),
    embedder,
    liveness,
    info: {
      detector: { id: manifest.detector.id, executionProvider: det.executionProvider },
      embedder: { modelKey: embedder.modelKey, executionProvider: emb.executionProvider, fallbackReason: emb.fallbackReason },
      liveness: { models: liveness.map((l) => l.manifest.id), missing },
    },
  };
}

export { OnnxLivenessChecker } from './liveness.js';
export * from './types.js';
