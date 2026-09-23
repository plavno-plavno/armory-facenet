import { randomBytes } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import pino from 'pino';
import { FaceIdClient } from '@faceid/shared';
import { Engine } from '../src/engine.js';
import { buildServer } from '../src/server/app.js';

export const ROOT = path.resolve(__dirname, '../../..');
export const MODELS = path.join(ROOT, 'models');
export const FIX = path.join(ROOT, 'tests/fixtures');

export const enrollPhoto = (name: string) => new Blob([readFileSync(path.join(FIX, 'enroll', name))], { type: 'image/jpeg' });
export const fixtureBlob = (rel: string, type = 'image/jpeg') => new Blob([readFileSync(path.join(FIX, rel))], { type });

/** Thresholds suitable for the small LFW-based fixtures (the spec defaults are placeholders, §7.7). */
export const TEST_CONFIG = {
  server: { enrollRatePerSec: 1000 },
  enroll: { minShortSide: 200, quality: { maxYaw: 0.25, minSharpness: 50 } },
  quality: { minSharpness: 30 },
  match: { acceptThreshold: 0.35, rejectThreshold: 0.2, margin: 0.05 },
  // Some low-res LFW press photos score as spoofs; liveness has its own tests.
  liveness: { enabled: false },
};

export interface TestEngine {
  engine: Engine;
  client: FaceIdClient;
  baseUrl: string;
  token: string;
  dataDir: string;
  dek: Buffer;
  close(opts?: { keepData?: boolean }): Promise<void>;
}

export async function startEngine(opts: { dataDir?: string; dek?: Buffer; config?: Record<string, unknown>; logs?: unknown[] } = {}): Promise<TestEngine> {
  const dataDir = opts.dataDir ?? mkdtempSync(path.join(tmpdir(), 'faceid-test-'));
  const dek = opts.dek ?? randomBytes(32);
  const sink = opts.logs;
  const logger = pino({ level: 'info' }, { write: (s: string) => void sink?.push(s) });
  // Test settings are persisted like user settings (so PATCH /config behaves normally);
  // only the port and execution provider are runtime overrides.
  mkdirSync(dataDir, { recursive: true });
  const cfgFile = path.join(dataDir, 'config.json');
  if (!existsSync(cfgFile)) writeFileSync(cfgFile, JSON.stringify({ ...TEST_CONFIG, ...(opts.config ?? {}) }));
  const engine = await Engine.create({
    dataDir,
    modelsDir: MODELS,
    dek,
    // Limit ORT threads: test files run in parallel and would otherwise oversubscribe the CPU.
    configOverrides: { server: { port: 0 }, models: { executionProvider: 'cpu', intraOpThreads: 4 } },
    logger,
  });
  await engine.start();
  const app = await buildServer(engine);
  await app.listen({ host: '127.0.0.1', port: 0 });
  const port = (app.server.address() as { port: number }).port;
  const baseUrl = `http://127.0.0.1:${port}/api/v1`;
  const token = engine.tokens.current();
  return {
    engine,
    client: new FaceIdClient({ baseUrl, token }),
    baseUrl,
    token,
    dataDir,
    dek,
    async close({ keepData = false } = {}) {
      await app.close();
      await engine.close();
      if (!keepData) rmSync(dataDir, { recursive: true, force: true });
    },
  };
}

export const consent = { obtained: true, basis: 'test' };

export async function waitFor<T>(fn: () => T | undefined | null | false, timeoutMs = 30000, stepMs = 100): Promise<T> {
  const t0 = Date.now();
  for (;;) {
    const v = fn();
    if (v) return v;
    if (Date.now() - t0 > timeoutMs) throw new Error('waitFor timeout');
    await new Promise((r) => setTimeout(r, stepMs));
  }
}
