// Engine entry point.
// - Under Electron: forked with utilityProcess; main sends {type:'init'} with the DEK and the
//   webcam capture MessagePort (spec §4, §12.2, §14).
// - Standalone (dev / headless): key from FACEID_PASSWORD (scrypt keystore), webcams via ffmpeg.

import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Engine } from './engine.js';
import { PortWebcamProvider, type PortLike } from './pipeline/sources/webcam.js';
import { buildServer } from './server/app.js';
import { releaseStalePort } from './pipeline/sources/device-lock.js';
import { loadOrCreateDek, passwordWrapper } from './store/keystore.js';

interface InitMessage {
  type: 'init';
  dataDir: string;
  modelsDir: string;
  dek: Uint8Array | null;
  configOverrides?: Record<string, unknown>;
  uiDir?: string;
}

interface ParentPort {
  on(event: 'message', cb: (e: { data: any; ports: PortLike[] }) => void): void;
  postMessage(msg: unknown): void;
}

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, '../../..');

export async function runEngine(init: InitMessage, capturePort: PortLike | null, notify: (msg: unknown) => void) {
  const engine = await Engine.create({
    dataDir: init.dataDir,
    modelsDir: init.modelsDir,
    dek: init.dek ? Buffer.from(init.dek) : undefined,
    configOverrides: init.configOverrides,
    webcams: capturePort ? new PortWebcamProvider(capturePort) : undefined,
  });
  await engine.start();
  const app = await buildServer(engine, { uiDir: init.uiDir });
  const { host, port } = engine.cfg().server;
  if (port) {
    // A previous engine suspended with Ctrl+Z still owns the port (and its cameras): stop it first.
    const holder = await releaseStalePort(port, (m) => engine.log.warn(m));
    if (holder) throw new Error(`Port ${port} is in use by ${holder}. Stop it or set FACEID_PORT.`);
  }
  await app.listen({ host, port });
  engine.log.info({ host, port }, 'API listening');

  const rotate = engine.tokens.rotate.bind(engine.tokens);
  engine.tokens.rotate = async () => {
    const t = await rotate();
    notify({ type: 'token', token: t });
    return t;
  };
  notify({ type: 'ready', host, port, token: engine.tokens.current(), tls: !!engine.cfg().server.tls });

  let closing = false;
  const shutdown = async () => {
    if (closing) return;
    closing = true;
    engine.log.info('shutting down');
    await app.close().catch(() => undefined);
    await engine.close().catch(() => undefined);
    process.exit(0);
  };
  return { engine, app, shutdown };
}

async function main() {
  const parentPort = (process as unknown as { parentPort?: ParentPort }).parentPort;
  if (parentPort) {
    // utilityProcess mode
    parentPort.on('message', async (e) => {
      const msg = e.data;
      if (msg?.type === 'init') {
        try {
          const { shutdown } = await runEngine(msg as InitMessage, e.ports?.[0] ?? null, (m) => parentPort.postMessage(m));
          parentPort.on('message', (ev) => {
            if (ev.data?.type === 'shutdown') void shutdown();
          });
        } catch (err) {
          parentPort.postMessage({ type: 'fatal', message: (err as Error).message });
          process.exit(1);
        }
      }
    });
    return;
  }

  // standalone mode
  const dataDir = path.resolve(process.env.FACEID_DATA_DIR ?? path.join(repoRoot, '.data'));
  const modelsDir = path.resolve(process.env.FACEID_MODELS_DIR ?? path.join(repoRoot, 'models'));
  const plaintext = process.env.FACEID_DEV_PLAINTEXT === '1';
  let dek: Buffer | null = null;
  if (!plaintext) {
    const password = process.env.FACEID_PASSWORD;
    if (!password) {
      console.error('Set FACEID_PASSWORD (keystore password) or FACEID_DEV_PLAINTEXT=1 for unencrypted development mode.');
      process.exit(2);
    }
    dek = await loadOrCreateDek(dataDir, passwordWrapper(password));
  }
  const overrides: Record<string, unknown> = plaintext ? { privacy: { encryptAtRest: false } } : {};
  if (process.env.FACEID_PORT) overrides.server = { port: Number(process.env.FACEID_PORT) };
  // Web mode: serve the admin UI built by `npm run build:ui` (FACEID_UI_DIR overrides, FACEID_UI=0 disables).
  const uiCandidate = path.resolve(process.env.FACEID_UI_DIR ?? path.join(repoRoot, 'apps/electron/dist/admin-ui'));
  const uiDir = process.env.FACEID_UI !== '0' && existsSync(path.join(uiCandidate, 'index.html')) ? uiCandidate : undefined;
  const { shutdown } = await runEngine({ type: 'init', dataDir, modelsDir, dek, configOverrides: overrides, uiDir }, null, (m: any) => {
    if (m.type !== 'ready') return;
    const base = `http${m.tls ? 's' : ''}://${m.host}:${m.port}`;
    console.log(`FaceID Engine API: ${base}/api/v1`);
    if (uiDir) console.log(`Admin UI:          ${base}/`);
    console.log(`API token:         ${m.token}`);
  });
  process.on('SIGINT', () => void shutdown());
  process.on('SIGTERM', () => void shutdown());
}

const isEntry =
  !!(process as unknown as { parentPort?: unknown }).parentPort ||
  (!!process.argv[1] && fileURLToPath(import.meta.url) === path.resolve(process.argv[1]));
if (isEntry) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
