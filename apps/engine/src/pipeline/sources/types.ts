import type { SourceState } from '@faceid/shared';
import type { Frame } from '../../vision/types.js';

export interface SourceStatus {
  state: SourceState;
  message?: string;
}

export interface FrameSource {
  readonly id: string;
  start(): Promise<void>;
  stop(): Promise<void>;
  onFrame(cb: (f: Frame) => void): void;
  onStatus(cb: (s: SourceStatus) => void): void;
}

export interface CameraDevice {
  deviceId: string;
  label: string;
}

/** Supplies webcam frames: the Electron capture window, or ffmpeg when running standalone. */
export interface WebcamProvider {
  listDevices(): Promise<CameraDevice[]>;
  createSource(id: string, deviceId: string | undefined, opts: { width?: number; height?: number; fps: number }): FrameSource;
}

/** Small helper so sources share listener plumbing. */
export abstract class BaseSource implements FrameSource {
  private frameCbs: ((f: Frame) => void)[] = [];
  private statusCbs: ((s: SourceStatus) => void)[] = [];
  protected status: SourceStatus = { state: 'stopped' };

  constructor(readonly id: string) {}

  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;

  onFrame(cb: (f: Frame) => void): void {
    this.frameCbs.push(cb);
  }

  onStatus(cb: (s: SourceStatus) => void): void {
    this.statusCbs.push(cb);
  }

  protected emitFrame(f: Frame): void {
    for (const cb of this.frameCbs) cb(f);
  }

  protected setStatus(state: SourceStatus['state'], message?: string): void {
    if (this.status.state === state && this.status.message === message) return;
    this.status = { state, message };
    for (const cb of this.statusCbs) cb(this.status);
  }
}
