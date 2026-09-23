// Webcam providers (spec §7.2).
// - PortWebcamProvider: frames from the hidden Electron capture window (getUserMedia) over a MessagePort.
// - FfmpegWebcamProvider: standalone engine (dev / headless), ffmpeg reads the device directly.

import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { rgbaToBgr } from '../../vision/preprocess.js';
import { describeHolders, releaseDevice } from './device-lock.js';
import { FfmpegSource, ffmpegPath } from './ffmpeg.js';
import { BaseSource, type CameraDevice, type FrameSource, type WebcamProvider } from './types.js';

/** Pixel formats a V4L2 device offers (queried via ioctl; does not start streaming). Cached per device. */
const formatCache = new Map<string, string>();
function v4l2Formats(device: string): string {
  if (!formatCache.has(device)) {
    let out = '';
    try {
      execFileSync(ffmpegPath(), ['-hide_banner', '-f', 'v4l2', '-list_formats', 'all', '-i', device], { stdio: ['ignore', 'ignore', 'pipe'], timeout: 5000 });
    } catch (e) {
      out = String((e as { stderr?: Buffer }).stderr ?? ''); // ffmpeg prints the list to stderr and exits non-zero
    }
    formatCache.set(device, out);
  }
  return formatCache.get(device)!;
}

// ---------- Electron capture window over MessagePort ----------

export interface PortLike {
  postMessage(msg: unknown): void;
  on(event: 'message', cb: (e: { data: any }) => void): void;
  start?(): void;
}

export type CaptureToEngine =
  | { type: 'devices'; reqId: number; devices: CameraDevice[] }
  | { type: 'frame'; streamId: string; width: number; height: number; data: ArrayBuffer | Uint8Array }
  | { type: 'status'; streamId: string; state: 'starting' | 'running' | 'error' | 'stopped'; message?: string };

export type EngineToCapture =
  | { type: 'list'; reqId: number }
  | { type: 'open'; streamId: string; deviceId?: string; width?: number; height?: number; fps: number }
  | { type: 'close'; streamId: string };

class PortWebcamSource extends BaseSource {
  constructor(
    id: string,
    private readonly provider: PortWebcamProvider,
    private readonly deviceId: string | undefined,
    private readonly opts: { width?: number; height?: number; fps: number },
  ) {
    super(id);
  }

  async start(): Promise<void> {
    this.setStatus('starting');
    this.provider.attach(this);
    this.provider.send({ type: 'open', streamId: this.id, deviceId: this.deviceId, ...this.opts });
  }

  async stop(): Promise<void> {
    this.provider.send({ type: 'close', streamId: this.id });
    this.provider.detach(this.id);
    this.setStatus('stopped');
  }

  deliver(msg: CaptureToEngine): void {
    if (msg.type === 'frame') {
      const rgba = msg.data instanceof Uint8Array ? msg.data : new Uint8Array(msg.data);
      if (this.status.state !== 'running') this.setStatus('running');
      this.emitFrame({ data: rgbaToBgr(rgba, msg.width, msg.height), width: msg.width, height: msg.height, ts: performance.now(), sourceId: this.id });
    } else if (msg.type === 'status') {
      this.setStatus(msg.state, msg.message);
    }
  }
}

export class PortWebcamProvider implements WebcamProvider {
  private readonly sources = new Map<string, PortWebcamSource>();
  private readonly pending = new Map<number, (d: CameraDevice[]) => void>();
  private reqId = 0;

  constructor(private readonly port: PortLike) {
    port.on('message', (e) => this.onMessage(e.data as CaptureToEngine));
    port.start?.();
  }

  send(msg: EngineToCapture): void {
    this.port.postMessage(msg);
  }

  attach(s: PortWebcamSource): void {
    this.sources.set(s.id, s);
  }

  detach(id: string): void {
    this.sources.delete(id);
  }

  private onMessage(msg: CaptureToEngine): void {
    if (msg.type === 'devices') {
      this.pending.get(msg.reqId)?.(msg.devices);
      this.pending.delete(msg.reqId);
      return;
    }
    this.sources.get(msg.streamId)?.deliver(msg);
  }

  listDevices(): Promise<CameraDevice[]> {
    const reqId = ++this.reqId;
    return new Promise((resolve) => {
      const t = setTimeout(() => {
        this.pending.delete(reqId);
        resolve([]);
      }, 5000);
      this.pending.set(reqId, (d) => {
        clearTimeout(t);
        resolve(d);
      });
      this.send({ type: 'list', reqId });
    });
  }

  createSource(id: string, deviceId: string | undefined, opts: { width?: number; height?: number; fps: number }): FrameSource {
    return new PortWebcamSource(id, this, deviceId, opts);
  }
}

// ---------- standalone: ffmpeg reads the device ----------

export class FfmpegWebcamProvider implements WebcamProvider {
  constructor(private readonly log?: (msg: string) => void) {}

  async listDevices(): Promise<CameraDevice[]> {
    if (process.platform === 'linux') {
      const sys = '/sys/class/video4linux';
      if (!existsSync(sys)) return [];
      return readdirSync(sys)
        .sort()
        .map((n) => {
          let label = n;
          try {
            label = readFileSync(`${sys}/${n}/name`, 'utf8').trim();
          } catch {
            /* ignore */
          }
          return { deviceId: `/dev/${n}`, label };
        });
    }
    return [];
  }

  createSource(id: string, deviceId: string | undefined, opts: { width?: number; height?: number; fps: number }): FrameSource {
    let inputArgs: string[];
    let target: string;
    if (process.platform === 'win32') {
      inputArgs = ['-f', 'dshow'];
      target = `video=${deviceId ?? ''}`;
    } else if (process.platform === 'darwin') {
      inputArgs = ['-f', 'avfoundation', '-framerate', '30'];
      target = `${deviceId ?? '0'}:none`;
    } else {
      target = deviceId ?? '/dev/video0';
      // Raw YUYV at HD saturates USB (typically 5–10 fps); MJPEG gives the camera's full frame rate.
      const width = opts.width ?? 1280;
      const height = opts.height ?? 720;
      inputArgs = ['-f', 'v4l2'];
      if (/Compressed\s*:\s*mjpeg/i.test(v4l2Formats(target))) inputArgs.push('-input_format', 'mjpeg');
      inputArgs.push('-framerate', '30', '-video_size', `${width}x${height}`);
      // Before each open: stop stale ffmpegs of earlier engine runs; report other applications.
      const prepare = async () => {
        const holders = await releaseDevice(target, this.log);
        return holders.length ? `camera is busy: ${describeHolders(holders)}` : null;
      };
      return new FfmpegSource(id, { kind: 'device', target, inputArgs, prepare }, { fps: opts.fps, width, height });
    }
    return new FfmpegSource(id, { kind: 'device', target, inputArgs }, { fps: opts.fps, width: opts.width, height: opts.height });
  }
}
