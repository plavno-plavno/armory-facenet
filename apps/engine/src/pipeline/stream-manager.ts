// Stream definitions (persisted encrypted: RTSP URLs may carry credentials), lifecycle, snapshots,
// and camera capture for enrollment (spec §9.3, §10.5).

import { existsSync, promises as fs } from 'node:fs';
import path from 'node:path';
import type { RecognitionEvent, StreamInfo, StreamInput } from '@faceid/shared';
import { enrollQuality } from '../config/schema.js';
import type { AnalyzedPhoto, FaceAnalyzer } from '../core/analyzer.js';
import { writeFileAtomic } from '../store/atomic.js';
import { meanEmbedding } from '../vision/embedder.js';
import { identifyBurst } from './decision.js';
import { ApiError } from '../util/errors.js';
import { newId, nowIso } from '../util/ids.js';
import { encodeJpegBgr, encodePngRgb } from '../vision/image.js';
import { DefaultQualityAssessor } from '../vision/quality.js';
import type { Detection, Frame } from '../vision/types.js';
import { Burst } from './burst.js';
import { FfmpegSource } from './sources/ffmpeg.js';
import type { FrameSource, WebcamProvider } from './sources/types.js';
import { StreamRuntime, type RuntimeDeps } from './stream-runtime.js';

interface StoredStream extends StreamInput {
  id: string;
}

export class StreamManager {
  private readonly defs = new Map<string, StoredStream>();
  private readonly runtimes = new Map<string, StreamRuntime>();
  private readonly file: string;

  constructor(
    dataDir: string,
    private readonly deps: RuntimeDeps,
    private readonly webcams: WebcamProvider,
    private readonly analyzer: FaceAnalyzer,
  ) {
    this.file = path.join(dataDir, 'streams.json.enc');
  }

  async init(): Promise<void> {
    if (existsSync(this.file)) {
      const list = JSON.parse(this.deps.cipher.decrypt(await fs.readFile(this.file)).toString('utf8')) as StoredStream[];
      for (const s of list) this.defs.set(s.id, s);
    }
    for (const s of this.defs.values()) {
      if (s.enabled) {
        await this.start(s.id).catch((e) => this.deps.log.error({ sourceId: s.id, err: (e as Error).message }, 'stream start failed'));
      }
    }
  }

  private async persist(): Promise<void> {
    await writeFileAtomic(this.file, this.deps.cipher.encrypt(Buffer.from(JSON.stringify([...this.defs.values()]))));
  }

  private validate(s: StreamInput): void {
    if (s.type === 'rtsp' && !/^rtsps?:\/\//i.test(s.url ?? '')) throw new ApiError('VALIDATION_ERROR', 'rtsp stream requires url rtsp://...');
    if (s.type === 'file' && !s.url) throw new ApiError('VALIDATION_ERROR', 'file stream requires url (path)');
  }

  private makeSource(s: StoredStream): FrameSource {
    const fps = s.detectFps ?? this.deps.cfg().pipeline.detectFps;
    const res = s.resolution;
    if (s.type === 'webcam') return this.webcams.createSource(s.id, s.deviceId, { width: res?.width, height: res?.height, fps });
    return new FfmpegSource(
      s.id,
      { kind: s.type, target: s.url!, loop: s.loop },
      { fps, width: res?.width, height: res?.height },
    );
  }

  info(id: string): StreamInfo {
    const s = this.defs.get(id);
    if (!s) throw new ApiError('NOT_FOUND', 'Stream not found');
    const rt = this.runtimes.get(id);
    return {
      ...s,
      url: redactUrl(s.url),
      status: rt?.status.state ?? 'stopped',
      statusMessage: rt?.status.message,
      fps: Math.round((rt?.fps ?? 0) * 10) / 10,
      droppedFrames: rt?.dropped ?? 0,
      activeTracks: rt?.activeTracks() ?? 0,
    };
  }

  list(): StreamInfo[] {
    return [...this.defs.keys()].map((id) => this.info(id));
  }

  listCameras() {
    return this.webcams.listDevices();
  }

  runtimesList(): StreamRuntime[] {
    return [...this.runtimes.values()];
  }

  async create(input: StreamInput): Promise<StreamInfo> {
    this.validate(input);
    const s: StoredStream = { ...input, id: newId() };
    this.defs.set(s.id, s);
    await this.persist();
    if (s.enabled) await this.start(s.id);
    return this.info(s.id);
  }

  async patch(id: string, patch: Partial<StreamInput>): Promise<StreamInfo> {
    const cur = this.defs.get(id);
    if (!cur) throw new ApiError('NOT_FOUND', 'Stream not found');
    const next: StoredStream = { ...cur, ...patch, id };
    this.validate(next);
    const wasRunning = this.runtimes.has(id);
    if (wasRunning) await this.stop(id, false);
    this.defs.set(id, next);
    await this.persist();
    if (next.enabled && (wasRunning || patch.enabled)) await this.start(id);
    return this.info(id);
  }

  async remove(id: string): Promise<void> {
    if (!this.defs.has(id)) throw new ApiError('NOT_FOUND', 'Stream not found');
    await this.stop(id, false);
    this.defs.delete(id);
    await this.persist();
  }

  async start(id: string): Promise<StreamInfo> {
    const s = this.defs.get(id);
    if (!s) throw new ApiError('NOT_FOUND', 'Stream not found');
    if (!this.runtimes.has(id)) {
      const rt = new StreamRuntime(id, s, this.makeSource(s), this.deps);
      this.runtimes.set(id, rt);
      await rt.start();
    }
    if (!s.enabled) {
      s.enabled = true;
      await this.persist();
    }
    return this.info(id);
  }

  async stop(id: string, persist = true): Promise<StreamInfo | void> {
    const s = this.defs.get(id);
    if (!s) throw new ApiError('NOT_FOUND', 'Stream not found');
    const rt = this.runtimes.get(id);
    this.runtimes.delete(id);
    if (rt) await rt.stop();
    if (persist) {
      s.enabled = false;
      await this.persist();
      return this.info(id);
    }
  }

  async stopAll(): Promise<void> {
    await Promise.allSettled([...this.runtimes.values()].map((r) => r.stop()));
    this.runtimes.clear();
  }

  private running(id: string): StreamRuntime {
    if (!this.defs.has(id)) throw new ApiError('NOT_FOUND', 'Stream not found');
    const rt = this.runtimes.get(id);
    if (!rt || rt.status.state !== 'running') throw new ApiError('SOURCE_UNAVAILABLE', 'Stream is not running');
    return rt;
  }

  async snapshot(id: string, overlay: boolean, maxWidth?: number): Promise<Buffer> {
    const rt = this.running(id);
    const f = rt.lastFrame;
    if (!f) throw new ApiError('SOURCE_UNAVAILABLE', 'No frame received yet');
    let data = f.data;
    if (overlay) {
      data = Uint8Array.from(f.data);
      for (const d of rt.lastDets) drawDetection(data, f.width, f.height, d);
      const roi = rt.def.roi;
      if (roi) drawRect(data, f.width, f.height, roi.x, roi.y, roi.w, roi.h, [255, 200, 0], 1);
    }
    return encodeJpegBgr(data, f.width, f.height, maxWidth ? 75 : 85, maxWidth);
  }

  /**
   * On-demand recognition ("Recognize" button): collect a burst of the largest face with stream
   * quality thresholds (§7.5), embed the best frames and decide (§7.6–7.7). The result is also
   * published as a regular recognition.result event (trackId "manual-…") for the journal/webhooks.
   */
  async recognizeOnce(id: string, timeoutMs = 6000) {
    const rt = this.running(id);
    const cfg = this.deps.cfg();
    const assessor = new DefaultQualityAssessor(cfg.quality);
    const started = Date.now();
    const burst = await new Promise<Burst>((resolve, reject) => {
      let b: Burst | null = null;
      const finish = (err: ApiError | null) => {
        clearTimeout(timer);
        off();
        if (err) reject(err);
        else resolve(b!);
      };
      const timer = setTimeout(() => {
        if (!b) return finish(new ApiError('NO_FACE', 'No face in front of the camera'));
        if (!b.enoughGood(cfg.burst)) {
          return finish(new ApiError('LOW_QUALITY', 'Not enough good frames', { reasons: Object.keys(b.reasons), counts: b.reasons, good: b.good.length }));
        }
        finish(null);
      }, timeoutMs);
      const off = rt.addListener((frame, dets) => {
        if (!dets.length) return;
        // Cooperative scenario: the person in front of the camera is the largest face.
        const det = dets.reduce((a, c) => (c.box[2] * c.box[3] > a.box[2] * a.box[3] ? c : a));
        b ??= new Burst(frame.ts);
        const aligned = this.deps.vision.aligner.align(frame, det.landmarks);
        const quality = assessor.assess(frame, det, aligned);
        b.add({ det, aligned, quality, frame: quality.passed ? frame : undefined });
        if (b.isComplete(frame.ts, cfg.burst) && b.enoughGood(cfg.burst)) finish(null);
      });
    });

    const best = burst.best(cfg.burst.topK);
    const embs = await this.deps.vision.embedder.embed(best.map((c) => c.aligned));
    const index = this.deps.index();
    const d = identifyBurst(index, embs, cfg.match);
    const probeCands = index.match(meanEmbedding(embs), 3);
    const r4 = (v: number | null) => (v === null ? null : Math.round(v * 10000) / 10000);
    const top = d.status === 'match' && d.top ? this.deps.person(d.top.personId) : undefined;
    const ev: RecognitionEvent = {
      type: 'recognition.result',
      eventId: newId(),
      ts: nowIso(),
      sourceId: id,
      trackId: `manual-${Date.now().toString(36)}`,
      status: d.status,
      score: r4(d.score),
      secondScore: r4(d.secondScore),
      frameAgreement: r4(d.frameAgreement),
      framesUsed: embs.length,
      attempt: 1,
      latencyMs: Date.now() - started,
      liveness: null,
    };
    if (top) {
      ev.person = { id: top.id, externalId: top.externalId, firstName: top.firstName, lastName: top.lastName };
      ev.personId = top.id;
    }
    this.deps.bus.publish(ev);
    return {
      event: ev,
      face: (await encodePngRgb(best[0].aligned.rgb, 112, 112)).toString('base64'),
      faceBox: best[0].det.box.map((v) => Math.round(v)),
      candidates: probeCands.map((c) => {
        const p = this.deps.person(c.personId);
        return { personId: c.personId, name: p ? `${p.lastName} ${p.firstName}` : c.personId, score: r4(c.score)!, photoId: c.photoId };
      }),
      framesSeen: burst.considered,
      goodFrames: burst.good.length,
    };
  }

  /**
   * Enrollment from camera (spec §9.3): wait for exactly one face, collect a burst with
   * enroll thresholds, return the best `enroll.captureFrames` frames analyzed for enrollment.
   */
  async capture(id: string): Promise<AnalyzedPhoto[]> {
    const rt = this.running(id);
    const cfg = this.deps.cfg();
    const assessor = new DefaultQualityAssessor(enrollQuality(cfg));
    const need = cfg.enroll.captureFrames;
    const burstCfg = { ...cfg.burst, maxFrames: Math.max(cfg.burst.maxFrames, need * 3) };

    const chosen = await new Promise<{ frame: Frame; det: Detection }[]>((resolve, reject) => {
      let burst: Burst | null = null;
      let lastReasons: Record<string, number> = {};
      const finish = (err: ApiError | null, value?: { frame: Frame; det: Detection }[]) => {
        clearTimeout(timer);
        off();
        if (err) reject(err);
        else resolve(value!);
      };
      const timer = setTimeout(() => {
        if (burst && burst.good.length > 0) {
          const best = burst.best(need);
          return finish(null, best.map((c) => ({ frame: c.frame!, det: c.det })));
        }
        finish(
          burst
            ? new ApiError('LOW_QUALITY', 'No frame of sufficient quality captured', { reasons: Object.keys(lastReasons) })
            : new ApiError('NO_FACE', 'No face appeared in front of the camera'),
        );
      }, cfg.enroll.captureTimeoutMs);
      const off = rt.addListener((frame, dets) => {
        if (dets.length > 1) return finish(new ApiError('MULTIPLE_FACES', 'More than one face in frame', { boxes: dets.map((d) => d.box.map(Math.round)) }));
        if (dets.length === 0) return;
        const det = dets[0];
        burst ??= new Burst(frame.ts);
        const aligned = this.deps.vision.aligner.align(frame, det.landmarks);
        const quality = assessor.assess(frame, det, aligned);
        // Keep the full frame only for candidates that passed (bounded by the burst size).
        burst.add({ det, aligned, quality, frame: quality.passed ? frame : undefined });
        lastReasons = burst.reasons;
        if (burst.good.length >= need && burst.isComplete(frame.ts, burstCfg)) {
          finish(null, burst.best(need).map((c) => ({ frame: c.frame!, det: c.det })));
        }
      });
    });

    const out: AnalyzedPhoto[] = [];
    for (const c of chosen) {
      const jpeg = await encodeJpegBgr(c.frame.data, c.frame.width, c.frame.height, 95);
      out.push(await this.analyzer.finishEnrollment(c.frame, c.det, jpeg));
    }
    return out;
  }
}

function redactUrl(url?: string): string | undefined {
  if (!url) return url;
  return url.replace(/(\w+:\/\/)([^@/]+)@/, (_m, p) => `${p}***@`);
}

function drawRect(buf: Uint8Array, w: number, h: number, x: number, y: number, rw: number, rh: number, bgr: number[], t = 2): void {
  const x0 = Math.max(0, Math.round(x));
  const y0 = Math.max(0, Math.round(y));
  const x1 = Math.min(w - 1, Math.round(x + rw));
  const y1 = Math.min(h - 1, Math.round(y + rh));
  const put = (px: number, py: number) => {
    if (px < 0 || py < 0 || px >= w || py >= h) return;
    const o = (py * w + px) * 3;
    buf[o] = bgr[0];
    buf[o + 1] = bgr[1];
    buf[o + 2] = bgr[2];
  };
  for (let k = 0; k < t; k++) {
    for (let px = x0; px <= x1; px++) {
      put(px, y0 + k);
      put(px, y1 - k);
    }
    for (let py = y0; py <= y1; py++) {
      put(x0 + k, py);
      put(x1 - k, py);
    }
  }
}

function drawDetection(buf: Uint8Array, w: number, h: number, d: Detection): void {
  drawRect(buf, w, h, d.box[0], d.box[1], d.box[2], d.box[3], [0, 255, 0]);
  const colors = [[255, 0, 0], [0, 0, 255], [0, 255, 255], [255, 0, 255], [255, 255, 0]];
  d.landmarks.forEach(([lx, ly], i) => drawRect(buf, w, h, lx - 2, ly - 2, 4, 4, colors[i], 2));
}
