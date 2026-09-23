// Model manifest (spec §6.4), sha256 verification and ONNX Runtime session creation.

import { createHash } from 'node:crypto';
import { createReadStream, readFileSync } from 'node:fs';
import path from 'node:path';
import * as ort from 'onnxruntime-node';
import { prepareCuda } from './cuda.js';

export interface DetectorManifest {
  id: string;
  file: string;
  sha256: string;
  channelOrder: 'BGR';
  normalize: null;
}

export interface EmbedderManifest {
  id: string;
  file: string;
  sha256: string;
  inputSize: [number, number];
  channelOrder: 'RGB' | 'BGR';
  mean: [number, number, number];
  std: [number, number, number];
  embeddingDim: number;
  dynamicBatch: boolean;
  preprocessVersion: number;
}

export interface ModelManifest {
  detector: DetectorManifest;
  embedder: EmbedderManifest;
  /** Alternative embedders selectable via config `models.embedder`. */
  embedders?: EmbedderManifest[];
}

export type ExecutionProviderPref = 'auto' | 'cpu' | 'cuda' | 'dml';

export function modelKeyOf(m: EmbedderManifest): string {
  return `${m.id}@${m.preprocessVersion}`;
}

export function loadManifest(modelsDir: string): ModelManifest {
  const raw = readFileSync(path.join(modelsDir, 'manifest.json'), 'utf8');
  return JSON.parse(raw) as ModelManifest;
}

/** Pick the embedder entry by id (falls back to the default `embedder`). */
export function selectEmbedder(manifest: ModelManifest, id?: string): EmbedderManifest {
  if (!id || id === manifest.embedder.id) return manifest.embedder;
  const found = manifest.embedders?.find((e) => e.id === id);
  if (!found) throw new Error(`Embedder "${id}" is not listed in manifest`);
  return found;
}

export async function sha256File(file: string): Promise<string> {
  const h = createHash('sha256');
  await new Promise<void>((resolve, reject) => {
    createReadStream(file)
      .on('data', (c) => h.update(c))
      .on('end', () => resolve())
      .on('error', reject);
  });
  return h.digest('hex');
}

export async function verifyModelFile(modelsDir: string, entry: { file: string; sha256: string }): Promise<string> {
  const file = path.join(modelsDir, entry.file);
  const actual = await sha256File(file);
  if (actual !== entry.sha256) {
    throw new Error(`Model checksum mismatch for ${entry.file}: expected ${entry.sha256}, got ${actual}`);
  }
  return file;
}

function candidateProviders(pref: ExecutionProviderPref): string[] {
  if (pref === 'cpu') return ['cpu'];
  if (pref === 'cuda') return ['cuda'];
  if (pref === 'dml') return ['dml'];
  if (process.platform === 'win32') return ['dml', 'cpu'];
  // With an NVIDIA driver present always try CUDA first (its runtime is preloaded if shipped).
  if (process.platform === 'linux' && process.arch === 'x64') return prepareCuda().gpu ? ['cuda', 'cpu'] : ['cpu'];
  return ['cpu'];
}

export interface LoadedSession {
  session: ort.InferenceSession;
  executionProvider: string;
  /** Why a preferred GPU provider was not used (null when it was, or when CPU was requested). */
  fallbackReason: string | null;
}

/**
 * Create a session trying GPU providers first when `auto`, falling back to CPU.
 */
export async function createSession(
  file: string,
  pref: ExecutionProviderPref,
  intraOpThreads = 0,
): Promise<LoadedSession> {
  const errors: string[] = [];
  const candidates = candidateProviders(pref);
  if (candidates.includes('cuda')) prepareCuda();
  for (const ep of candidates) {
    try {
      const opts: ort.InferenceSession.SessionOptions = {
        executionProviders: [ep],
        graphOptimizationLevel: 'all',
        // 3 = errors only: hides ORT's informational "nodes not assigned to the preferred EP" warning
        // (shape ops intentionally stay on CPU with the CUDA provider).
        logSeverityLevel: 3,
      };
      if (intraOpThreads > 0) opts.intraOpNumThreads = intraOpThreads;
      const session = await ort.InferenceSession.create(file, opts);
      return { session, executionProvider: ep, fallbackReason: errors.length ? errors.join('; ') : null };
    } catch (e) {
      errors.push(`${ep}: ${(e as Error).message}`);
    }
  }
  throw new Error(`Could not create ONNX session for ${path.basename(file)}: ${errors.join('; ')}`);
}
