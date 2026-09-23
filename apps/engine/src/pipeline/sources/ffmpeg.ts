// RTSP / file / local camera frames via an ffmpeg child process (spec §7.2):
// ffmpeg ... -vf scale=W:H -fpsmax <detectFps> -f rawvideo -pix_fmt bgr24 -   (files: -vf fps=<detectFps>,scale=W:H)

import { spawn, type ChildProcess } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { BaseSource } from './types.js';

export interface FfmpegInput {
  kind: 'rtsp' | 'file' | 'device';
  /** URL, file path, or device spec. */
  target: string;
  /** Extra args before -i (device formats). */
  inputArgs?: string[];
  loop?: boolean;
  /** Runs before every (re)launch, e.g. to free a camera; a returned string is shown as the status reason. */
  prepare?: () => Promise<string | null>;
}

export interface FfmpegSourceOptions {
  fps: number;
  width?: number;
  height?: number;
  maxLongSide?: number;
  stallTimeoutMs?: number;
  realtime?: boolean; // -re for files (default true)
}

export function ffmpegPath(): string {
  if (process.env.FACEID_FFMPEG) return process.env.FACEID_FFMPEG;
  const res = (process as { resourcesPath?: string }).resourcesPath;
  if (res) {
    const bin = path.join(res, 'ffmpeg', process.platform === 'win32' ? 'ffmpeg.exe' : 'ffmpeg');
    if (existsSync(bin)) return bin;
  }
  return 'ffmpeg';
}

export function ffprobePath(): string {
  if (process.env.FACEID_FFPROBE) return process.env.FACEID_FFPROBE;
  const ff = ffmpegPath();
  return ff === 'ffmpeg' ? 'ffprobe' : ff.replace(/ffmpeg(\.exe)?$/, (m, ext) => `ffprobe${ext ?? ''}`);
}

/** Probe the native frame size of an input. */
export function probeSize(input: FfmpegInput, timeoutMs = 15000): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const args = ['-v', 'error', ...(input.kind === 'rtsp' ? ['-rtsp_transport', 'tcp'] : []), ...(input.inputArgs ?? [])];
    args.push('-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', input.target);
    const p = spawn(ffprobePath(), args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '';
    let err = '';
    const timer = setTimeout(() => {
      p.kill('SIGKILL');
      reject(new Error('ffprobe timeout'));
    }, timeoutMs);
    p.stdout.on('data', (c) => (out += c));
    p.stderr.on('data', (c) => (err += c));
    p.on('error', (e) => {
      clearTimeout(timer);
      reject(e);
    });
    p.on('close', () => {
      clearTimeout(timer);
      try {
        const s = JSON.parse(out).streams?.[0];
        if (s?.width && s?.height) resolve({ width: s.width, height: s.height });
        else reject(new Error(err.trim() || 'no video stream'));
      } catch {
        reject(new Error(err.trim() || 'ffprobe failed'));
      }
    });
  });
}

/** Fit into maxLongSide keeping aspect; even dimensions for ffmpeg scalers. */
export function fitSize(w: number, h: number, maxLong: number): { width: number; height: number } {
  const s = Math.min(1, maxLong / Math.max(w, h));
  const even = (v: number) => Math.max(2, Math.round((v * s) / 2) * 2);
  return { width: even(w), height: even(h) };
}

export class FfmpegSource extends BaseSource {
  private proc: ChildProcess | null = null;
  private stopped = true;
  private backoffMs = 1000;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private stallTimer: NodeJS.Timeout | null = null;
  private size: { width: number; height: number } | null = null;

  constructor(
    id: string,
    private readonly input: FfmpegInput,
    private readonly opts: FfmpegSourceOptions,
  ) {
    super(id);
  }

  async start(): Promise<void> {
    if (!this.stopped) return;
    this.stopped = false;
    this.backoffMs = 1000;
    this.setStatus('starting');
    await this.launch();
  }

  async stop(): Promise<void> {
    this.stopped = true;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    if (this.stallTimer) clearTimeout(this.stallTimer);
    const p = this.proc;
    this.proc = null;
    if (p && p.exitCode === null) {
      await new Promise<void>((resolve) => {
        const t = setTimeout(() => {
          p.kill('SIGKILL');
          resolve();
        }, 2000);
        p.once('close', () => {
          clearTimeout(t);
          resolve();
        });
        p.kill('SIGTERM');
      });
    }
    this.setStatus('stopped');
  }

  private scheduleReconnect(reason: string): void {
    if (this.stopped) return;
    this.setStatus('reconnecting', reason);
    const delay = this.backoffMs;
    this.backoffMs = Math.min(30_000, this.backoffMs * 2);
    this.reconnectTimer = setTimeout(() => void this.launch(), delay);
  }

  private async resolveSize(): Promise<{ width: number; height: number }> {
    if (this.opts.width && this.opts.height) return { width: this.opts.width, height: this.opts.height };
    if (!this.size) {
      const native = await probeSize(this.input);
      this.size = fitSize(native.width, native.height, this.opts.maxLongSide ?? 1280);
    }
    return this.size;
  }

  private async launch(): Promise<void> {
    if (this.stopped) return;
    let size: { width: number; height: number };
    try {
      size = await this.resolveSize();
    } catch (e) {
      if (this.input.kind === 'file') {
        this.stopped = true;
        this.setStatus('error', (e as Error).message);
      } else this.scheduleReconnect((e as Error).message);
      return;
    }
    if (this.stopped) return;
    if (this.input.prepare) {
      const blocked = await this.input.prepare().catch(() => null);
      if (this.stopped) return;
      if (blocked) {
        this.scheduleReconnect(blocked);
        return;
      }
    }
    const { width, height } = size;
    const args = ['-hide_banner', '-loglevel', 'error', '-nostdin'];
    if (this.input.kind === 'rtsp') args.push('-rtsp_transport', 'tcp', '-timeout', '10000000');
    if (this.input.kind === 'file') {
      if (this.opts.realtime !== false) args.push('-re');
      if (this.input.loop) args.push('-stream_loop', '-1');
    }
    args.push(...(this.input.inputArgs ?? []), '-i', this.input.target);
    // Live sources: -fpsmax only drops frames above detectFps; the fps filter would also pad a slower
    // camera with copies of old frames. Files keep the fps filter so a debug video is sampled evenly.
    const rate = this.input.kind === 'file' ? ['-vf', `fps=${this.opts.fps},scale=${width}:${height}`] : ['-vf', `scale=${width}:${height}`, '-fpsmax', String(this.opts.fps)];
    args.push('-an', ...rate, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-');

    const p = spawn(ffmpegPath(), args, { stdio: ['ignore', 'pipe', 'pipe'] });
    this.proc = p;
    const frameBytes = width * height * 3;
    let buf = Buffer.allocUnsafe(frameBytes);
    let filled = 0;
    let stderr = '';
    const armStall = () => {
      if (this.stallTimer) clearTimeout(this.stallTimer);
      this.stallTimer = setTimeout(() => {
        if (this.proc === p) p.kill('SIGKILL');
      }, this.opts.stallTimeoutMs ?? 15_000);
    };
    armStall();

    p.stdout!.on('data', (chunk: Buffer) => {
      let off = 0;
      while (off < chunk.length) {
        const n = Math.min(frameBytes - filled, chunk.length - off);
        chunk.copy(buf, filled, off, off + n);
        filled += n;
        off += n;
        if (filled === frameBytes) {
          const data = new Uint8Array(buf.buffer, buf.byteOffset, frameBytes);
          buf = Buffer.allocUnsafe(frameBytes);
          filled = 0;
          this.backoffMs = 1000;
          if (this.status.state !== 'running') this.setStatus('running');
          armStall();
          this.emitFrame({ data, width, height, ts: performance.now(), sourceId: this.id });
        }
      }
    });
    p.stderr!.on('data', (c) => {
      stderr = (stderr + c).slice(-2000);
    });
    p.on('error', (e) => {
      if (this.proc !== p) return;
      this.proc = null;
      this.scheduleReconnect(e.message);
    });
    p.on('close', (code) => {
      if (this.stallTimer) clearTimeout(this.stallTimer);
      if (this.proc !== p) return;
      this.proc = null;
      if (this.stopped) return;
      if (this.input.kind === 'file' && code === 0) {
        this.stopped = true;
        this.setStatus('stopped', 'end of file');
        return;
      }
      const msg = stderr.trim().split('\n').pop() || `ffmpeg exited with code ${code}`;
      if (this.input.kind === 'file') {
        this.stopped = true;
        this.setStatus('error', msg);
      } else this.scheduleReconnect(msg);
    });
  }
}
