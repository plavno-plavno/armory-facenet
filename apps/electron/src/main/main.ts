// Electron main process (spec §14): lifecycle, tray, safeStorage-wrapped DEK, engine watchdog,
// hidden webcam capture window, admin window. Performs no recognition work itself.

import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  Menu,
  MessageChannelMain,
  nativeImage,
  net,
  protocol,
  safeStorage,
  session,
  systemPreferences,
  Tray,
  utilityProcess,
  type UtilityProcess,
} from 'electron';
import { loadOrCreateDek, passwordWrapper, readKeystoreMode, type KeyWrapper } from '../../../engine/src/store/keystore.js';

if (process.env.FACEID_USER_DATA) app.setPath('userData', process.env.FACEID_USER_DATA);
const isDev = !app.isPackaged;
const resources = isDev ? path.resolve(__dirname, '../../..') : process.resourcesPath;
const modelsDir = isDev ? path.join(resources, 'models') : path.join(resources, 'models');
const engineEntry = path.join(__dirname, 'engine.mjs');
const dataDir = process.env.FACEID_DATA_DIR ?? path.join(app.getPath('userData'), 'data');
const UI_ORIGIN = 'faceid://admin';

// The admin UI is served from a privileged custom scheme so the engine can allow exactly this CORS origin.
protocol.registerSchemesAsPrivileged([{ scheme: 'faceid', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true } }]);

interface EngineConn {
  baseUrl: string;
  token: string;
}

let engine: UtilityProcess | null = null;
let conn: EngineConn | null = null;
let status: 'starting' | 'running' | 'crashed' | 'failed' = 'starting';
let lastError = '';
const restarts: number[] = [];
let quitting = false;
let dek: Buffer | null = null;
let captureWin: BrowserWindow | null = null;
let adminWin: BrowserWindow | null = null;
let tray: Tray | null = null;
// Smoke-test mode (CI, spec §14): render the admin UI off-screen, save a PNG, report health, quit.
const smokeShot = process.env.FACEID_SMOKE_SHOT;

// ---------- data key (spec §12.2) ----------

function safeStorageUsable(): boolean {
  if (!safeStorage.isEncryptionAvailable()) return false;
  // On Linux without a secret service Chromium falls back to a hard-coded key: not acceptable.
  if (process.platform === 'linux' && safeStorage.getSelectedStorageBackend?.() === 'basic_text') return false;
  return true;
}

const safeStorageWrapper: KeyWrapper = {
  mode: 'safeStorage',
  async wrap(k) {
    return { wrapped: safeStorage.encryptString(k.toString('base64')) };
  },
  async unwrap(w) {
    return Buffer.from(safeStorage.decryptString(w), 'base64');
  },
};

async function askPassword(create: boolean): Promise<string | null> {
  const win = new BrowserWindow({
    width: 420,
    height: 260,
    resizable: false,
    title: 'FaceID Engine — unlock',
    webPreferences: { preload: path.join(__dirname, 'preload-admin.cjs'), contextIsolation: true, sandbox: true },
  });
  await win.loadURL(`${UI_ORIGIN}/unlock.html?create=${create ? '1' : '0'}`);
  return new Promise((resolve) => {
    const onPw = (_e: unknown, pw: string) => {
      ipcMain.removeListener('unlock:password', onPw);
      win.close();
      resolve(pw || null);
    };
    ipcMain.on('unlock:password', onPw);
    win.on('closed', () => resolve(null));
  });
}

async function obtainDek(): Promise<Buffer | null> {
  mkdirSync(dataDir, { recursive: true });
  const mode = await readKeystoreMode(dataDir);
  // Headless / CI: password from the environment, no dialog, no OS keyring.
  if (process.env.FACEID_PASSWORD && mode !== 'safeStorage') return loadOrCreateDek(dataDir, passwordWrapper(process.env.FACEID_PASSWORD));
  if (mode === 'safeStorage' || (mode === null && safeStorageUsable())) return loadOrCreateDek(dataDir, safeStorageWrapper);
  // Password mode: Linux without libsecret, or chosen explicitly (spec §12.2).
  for (let attempt = 0; attempt < 3; attempt++) {
    const pw = await askPassword(mode === null);
    if (!pw) return null;
    try {
      return await loadOrCreateDek(dataDir, passwordWrapper(pw));
    } catch {
      await dialog.showMessageBox({ type: 'error', message: 'Wrong password' });
    }
  }
  return null;
}

// ---------- engine lifecycle & watchdog (spec §13 "Resilience") ----------

function startEngine(): void {
  status = 'starting';
  const { port1, port2 } = new MessageChannelMain();
  engine = utilityProcess.fork(engineEntry, [], {
    serviceName: 'FaceID Engine',
    stdio: 'pipe',
    env: { ...process.env, FACEID_FFMPEG: process.env.FACEID_FFMPEG ?? ffmpegBinary(), ...cudaEnv() },
  });
  engine.stdout?.on('data', (d) => process.stdout.write(d));
  engine.stderr?.on('data', (d) => process.stderr.write(d));
  const configOverrides: Record<string, unknown> = { server: { corsOrigins: [UI_ORIGIN] } };
  if (process.env.FACEID_PORT) (configOverrides.server as Record<string, unknown>).port = Number(process.env.FACEID_PORT);
  engine.postMessage({ type: 'init', dataDir, modelsDir, dek: new Uint8Array(dek!), configOverrides }, [port1]);
  captureWin?.webContents.postMessage('capture:port', null, [port2]);

  engine.on('message', (m: any) => {
    if (m?.type === 'ready') {
      status = 'running';
      conn = { baseUrl: `http${m.tls ? 's' : ''}://${m.host === '0.0.0.0' ? '127.0.0.1' : m.host}:${m.port}/api/v1`, token: m.token };
      adminWin?.webContents.send('engine:conn', conn);
      updateTray();
      if (smokeShot) void runSmoke();
    } else if (m?.type === 'token' && conn) {
      conn = { ...conn, token: m.token };
      adminWin?.webContents.send('engine:conn', conn);
    } else if (m?.type === 'fatal') {
      lastError = m.message;
    }
  });
  engine.on('exit', (code) => {
    engine = null;
    conn = null;
    if (quitting) return;
    const now = Date.now();
    restarts.push(now);
    while (restarts.length && now - restarts[0] > 60_000) restarts.shift();
    if (restarts.length > 5) {
      status = 'failed';
      updateTray();
      adminWin?.webContents.send('engine:status', { status, error: lastError || `engine exited with code ${code}` });
      return;
    }
    status = 'crashed';
    updateTray();
    setTimeout(startEngine, 1000);
  });
}

/** Dev: expose the locally installed CUDA runtime (.cuda-libs, `npm run cuda:install`) to the engine. */
function cudaEnv(): Record<string, string> {
  const base = path.join(resources, '.cuda-libs', 'nvidia');
  if (process.platform !== 'linux' || !existsSync(base)) return {};
  const dirs = ['cu13/lib', 'cudnn/lib'].map((d) => path.join(base, d)).filter((d) => existsSync(d));
  return dirs.length ? { LD_LIBRARY_PATH: [...dirs, process.env.LD_LIBRARY_PATH].filter(Boolean).join(':') } : {};
}

function ffmpegBinary(): string {
  const bin = path.join(resources, 'ffmpeg', process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg');
  return existsSync(bin) ? bin : 'ffmpeg';
}

// ---------- windows ----------

function createCaptureWindow(): void {
  captureWin = new BrowserWindow({
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload-capture.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: false,
    },
  });
  void captureWin.loadFile(path.join(__dirname, 'capture', 'capture.html'));
}

function openAdmin(): void {
  if (adminWin && !adminWin.isDestroyed()) {
    adminWin.show();
    adminWin.focus();
    return;
  }
  adminWin = new BrowserWindow({
    width: 1280,
    height: 820,
    show: !smokeShot,
    title: 'FaceID Engine',
    webPreferences: { preload: path.join(__dirname, 'preload-admin.cjs'), contextIsolation: true, nodeIntegration: false, sandbox: true },
  });
  void adminWin.loadURL(`${UI_ORIGIN}/index.html`);
  adminWin.on('close', (e) => {
    if (!quitting && tray) {
      e.preventDefault();
      adminWin?.hide();
    }
  });
}

function updateTray(): void {
  if (!tray) return;
  tray.setToolTip(`FaceID Engine — ${status}`);
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: `Engine: ${status}`, enabled: false },
      { label: 'Open admin', click: openAdmin },
      { type: 'separator' },
      { label: 'Restart engine', click: () => (engine ? engine.kill() : startEngine()) },
      { label: 'Quit', click: () => app.quit() },
    ]),
  );
}

async function runSmoke(): Promise<void> {
  const { writeFileSync } = await import('node:fs');
  const routes = (process.env.FACEID_SMOKE_ROUTES ?? 'dashboard').split(',');
  for (const r of routes) {
    await adminWin!.webContents.executeJavaScript(`location.hash = '#/${r}'`);
    await new Promise((res) => setTimeout(res, 2500));
    const img = await adminWin!.webContents.capturePage();
    writeFileSync(routes.length > 1 ? smokeShot!.replace(/\.png$/, `-${r.replace(/\W/g, '_')}.png`) : smokeShot!, img.toPNG());
  }
  const health = await (await fetch(`${conn!.baseUrl}/health`, { headers: { Authorization: `Bearer ${conn!.token}` } })).json();
  console.log(`SMOKE health=${health.status} embedder=${health.models.embedder.modelKey}/${health.models.embedder.executionProvider} gallery=${JSON.stringify(health.gallery)}`);
  app.quit();
}

// ---------- IPC for the admin renderer (token never leaves main except to our own renderer) ----------

const fromAdminUi = (e: Electron.IpcMainInvokeEvent) => e.senderFrame?.url.startsWith(`${UI_ORIGIN}/`) ?? false;
ipcMain.handle('engine:getConn', (e) => (fromAdminUi(e) ? conn : null));
ipcMain.handle('engine:getStatus', () => ({ status, error: lastError }));

// ---------- app ----------

const single = app.requestSingleInstanceLock();
if (!single) app.quit();
app.on('second-instance', openAdmin);

app.whenReady().then(async () => {
  const uiRoot = path.join(__dirname, 'admin-ui');
  protocol.handle('faceid', (req) => {
    const u = new URL(req.url);
    const file = path.normalize(path.join(uiRoot, decodeURIComponent(u.pathname)));
    if (u.host !== 'admin' || !file.startsWith(uiRoot)) return new Response('not found', { status: 404 });
    return net.fetch(pathToFileURL(file).toString());
  });
  // Only the capture window may use the camera; nothing may navigate away or open windows.
  session.defaultSession.setPermissionRequestHandler((wc, permission, cb) => cb(permission === 'media' && wc === captureWin?.webContents));
  session.defaultSession.setPermissionCheckHandler((wc, permission) => permission === 'media' && wc === captureWin?.webContents);
  app.on('web-contents-created', (_e, wc) => {
    wc.setWindowOpenHandler(() => ({ action: 'deny' }));
    wc.on('will-navigate', (ev) => ev.preventDefault());
  });
  if (process.platform === 'darwin') await systemPreferences.askForMediaAccess('camera').catch(() => false);

  dek = await obtainDek();
  if (!dek) {
    app.quit();
    return;
  }
  createCaptureWindow();
  await new Promise<void>((r) => captureWin!.webContents.once('did-finish-load', () => r()));
  startEngine();

  const iconPath = path.join(__dirname, 'admin-ui', 'icon.png');
  if (smokeShot) {
    openAdmin();
    return;
  }
  tray = new Tray(existsSync(iconPath) ? nativeImage.createFromBuffer(readFileSync(iconPath)) : nativeImage.createEmpty());
  tray.on('click', openAdmin);
  updateTray();
  if (!process.argv.includes('--hidden')) openAdmin();
});

app.on('window-all-closed', () => {
  // Keep running in the tray: recognition continues without the admin window.
});

app.on('before-quit', (e) => {
  if (quitting || !engine) {
    quitting = true;
    return;
  }
  // Let the engine stop streams and flush the gallery cache before exiting.
  e.preventDefault();
  quitting = true;
  const proc = engine;
  const done = () => app.quit();
  const t = setTimeout(() => {
    proc.kill();
    done();
  }, 5000);
  proc.once('exit', () => {
    clearTimeout(t);
    done();
  });
  proc.postMessage({ type: 'shutdown' });
});
