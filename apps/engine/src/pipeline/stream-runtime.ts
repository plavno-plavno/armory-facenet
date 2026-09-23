// Per-stream processing: backpressure, detection, tracking, bursts, identification (spec §7.2–§7.8).

import { promises as fs } from 'node:fs';
import path from 'node:path';
import type { RecognitionEvent, StreamInput } from '@faceid/shared';
import type { EngineConfig } from '../config/schema.js';
import type { EventBus } from '../core/events.js';
import type { RecognitionJournal } from '../core/journal.js';
import type { GalleryIndex } from '../gallery/index.js';
import type { Cipher } from '../store/crypto.js';
import { newId, nowIso } from '../util/ids.js';
import type { Logger } from '../util/logger.js';
import { encodeJpegBgr } from '../vision/image.js';
import type { VisionStack } from '../vision/index.js';
import { DefaultQualityAssessor } from '../vision/quality.js';
import type { Detection, Frame, LivenessChecker } from '../vision/types.js';
import { Burst, type BurstCandidate } from './burst.js';
import { identifyBurst } from './decision.js';
import { IouTracker, type Track } from './tracker.js';
import type { FrameSource, SourceStatus } from './sources/types.js';

export type TrackState = 'NEW' | 'COLLECTING' | 'IDENTIFYING' | 'RESOLVED' | 'COOLDOWN';

interface TrackCtx {
  track: Track;
  state: TrackState;
  attempt: number;
  burst: Burst | null;
  alive: boolean;
}

export interface PersonLookup {
  (id: string): { id: string; externalId?: string; firstName: string; lastName: string } | undefined;
}

export interface RuntimeDeps {
  vision: VisionStack;
  cfg: () => EngineConfig;
  index: () => GalleryIndex;
  person: PersonLookup;
  bus: EventBus;
  journal: RecognitionJournal;
  cipher: Cipher;
  liveness: LivenessChecker;
  metrics: PipelineMetrics;
  log: Logger;
}

/** Shared latency statistics for /health. */
export class PipelineMetrics {
  private readonly lat: number[] = [];
  record(latencyMs: number): void {
    this.lat.push(latencyMs);
    if (this.lat.length > 500) this.lat.shift();
  }
  percentile(p: number): number | null {
    if (this.lat.length === 0) return null;
    const s = [...this.lat].sort((a, b) => a - b);
    return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
  }
}

export type FrameListener = (frame: Frame, dets: Detection[]) => void;

const r4 = (v: number | null) => (v === null ? null : Math.round(v * 10000) / 10000);

export class StreamRuntime {
  private latest: Frame | null = null;
  private busy = false;
  private tracker: IouTracker;
  private readonly tracks = new Map<string, TrackCtx>();
  private readonly lastMatch = new Map<string, number>(); // personId -> wall ms
  private readonly listeners = new Set<FrameListener>();
  private readonly pending = new Set<Promise<void>>();
  lastFrame: Frame | null = null;
  lastDets: Detection[] = [];
  received = 0;
  processed = 0;
  dropped = 0;
  fps = 0;
  private fpsWindow: number[] = [];
  status: SourceStatus = { state: 'stopped' };

  constructor(
    readonly id: string,
    readonly def: StreamInput,
    private readonly source: FrameSource,
    private readonly deps: RuntimeDeps,
  ) {
    const t = deps.cfg().tracker;
    this.tracker = new IouTracker(t.iouThreshold, t.ttlMs);
    source.onFrame((f) => this.onFrame(f));
    source.onStatus((s) => {
      this.status = s;
      deps.bus.publish({ type: 'stream.status', ts: nowIso(), sourceId: id, status: s.state, message: s.message });
    });
  }

  get detectFps(): number {
    return this.def.detectFps ?? this.deps.cfg().pipeline.detectFps;
  }

  async start(): Promise<void> {
    await this.source.start();
  }

  async stop(): Promise<void> {
    await this.source.stop();
    this.latest = null;
    await Promise.allSettled([...this.pending]);
    this.tracker.clear();
    this.tracks.clear();
  }

  activeTracks(): number {
    return this.tracker.live().length;
  }

  addListener(fn: FrameListener): () => void {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  private onFrame(f: Frame): void {
    this.received++;
    if (this.latest) this.dropped++; // only the newest frame is kept
    this.latest = f;
    if (!this.busy) void this.loop();
  }

  private async loop(): Promise<void> {
    this.busy = true;
    try {
      while (this.latest) {
        const f = this.latest;
        this.latest = null;
        const t0 = performance.now();
        try {
          await this.processFrame(f);
        } catch (e) {
          this.deps.log.error({ sourceId: this.id, err: (e as Error).message }, 'frame processing failed');
        }
        this.tickFps();
        const wait = 1000 / this.detectFps - (performance.now() - t0);
        if (wait > 1) await new Promise((r) => setTimeout(r, wait));
      }
    } finally {
      this.busy = false;
    }
  }

  private tickFps(): void {
    const now = performance.now();
    this.processed++;
    this.fpsWindow.push(now);
    while (this.fpsWindow.length && now - this.fpsWindow[0] > 2000) this.fpsWindow.shift();
    this.fps = this.fpsWindow.length > 1 ? ((this.fpsWindow.length - 1) * 1000) / (now - this.fpsWindow[0]) : 0;
  }

  private inRoi(d: Detection): boolean {
    const roi = this.def.roi;
    if (!roi) return true;
    const cx = d.box[0] + d.box[2] / 2;
    const cy = d.box[1] + d.box[3] / 2;
    return cx >= roi.x && cy >= roi.y && cx <= roi.x + roi.w && cy <= roi.y + roi.h;
  }

  async processFrame(frame: Frame): Promise<void> {
    const cfg = this.deps.cfg();
    const dets = (await this.deps.vision.detector.detect(frame)).filter((d) => this.inRoi(d));
    this.lastFrame = frame;
    this.lastDets = dets;
    for (const l of this.listeners) l(frame, dets);

    const { updated, expired } = this.tracker.update(dets, frame.ts);
    for (const t of expired) {
      const ctx = this.tracks.get(t.id);
      if (ctx) ctx.alive = false;
      this.tracks.delete(t.id);
    }
    for (const t of updated) if (!this.tracks.has(t.id)) this.tracks.set(t.id, { track: t, state: 'NEW', attempt: 1, burst: null, alive: true });

    // Only the largest faces are recognized concurrently (spec §7.3).
    const busyCount = [...this.tracks.values()].filter((c) => c.state === 'IDENTIFYING').length;
    const slots = Math.max(0, cfg.pipeline.maxConcurrentTracks - busyCount);
    const eligible = updated
      .map((t) => this.tracks.get(t.id)!)
      .filter((c) => c.state === 'NEW' || c.state === 'COLLECTING')
      .sort((a, b) => Math.min(b.track.det.box[2], b.track.det.box[3]) - Math.min(a.track.det.box[2], a.track.det.box[3]))
      .slice(0, slots);

    const assessor = new DefaultQualityAssessor(cfg.quality);
    for (const ctx of eligible) {
      if (ctx.state === 'NEW' || !ctx.burst) {
        ctx.state = 'COLLECTING';
        ctx.burst = new Burst(frame.ts);
      }
      const det = ctx.track.det;
      const aligned = this.deps.vision.aligner.align(frame, det.landmarks);
      const quality = assessor.assess(frame, det, aligned);
      const cand: BurstCandidate = { det, aligned, quality };
      if (cfg.events.storeSnapshots) cand.frame = frame;
      ctx.burst.add(cand);
      if (ctx.burst.isComplete(frame.ts, cfg.burst)) this.finishBurst(ctx);
    }
  }

  private finishBurst(ctx: TrackCtx): void {
    const cfg = this.deps.cfg();
    const burst = ctx.burst!;
    ctx.burst = null;
    if (!burst.enoughGood(cfg.burst)) {
      void this.publish(ctx, { status: 'low_quality', framesUsed: burst.good.length, reasons: burst.reasons }, burst.best(1)[0]);
      if (ctx.attempt > cfg.pipeline.maxRetries) ctx.state = 'COOLDOWN';
      else {
        ctx.attempt++;
        ctx.state = 'COLLECTING';
      }
      return;
    }
    ctx.state = 'IDENTIFYING';
    const p = this.identify(ctx, burst.best(cfg.burst.topK)).catch((e) => {
      this.deps.log.error({ sourceId: this.id, trackId: ctx.track.id, err: (e as Error).message }, 'identification failed');
      ctx.state = 'COLLECTING';
    });
    this.pending.add(p);
    void p.finally(() => this.pending.delete(p));
  }

  private async identify(ctx: TrackCtx, best: BurstCandidate[]): Promise<void> {
    const cfg = this.deps.cfg();
    const embs = await this.deps.vision.embedder.embed(best.map((c) => c.aligned));
    // v1: NoopLivenessChecker; the result is not exposed (event.liveness = null, spec §7.9).
    await this.deps.liveness.check(best.map((c) => c.aligned), best.flatMap((c) => (c.frame ? [c.frame] : [])));
    const d = identifyBurst(this.deps.index(), embs, cfg.match);
    let status: RecognitionEvent['status'] = d.status;
    if (status === 'uncertain') {
      if (ctx.attempt <= cfg.pipeline.maxRetries) {
        await this.publish(ctx, { status, framesUsed: embs.length, decision: d }, best[0]);
        ctx.attempt++;
        ctx.state = 'COLLECTING';
        return;
      }
      status = 'unknown';
    }
    ctx.state = 'COOLDOWN';
    if (status === 'match' && d.top) {
      const now = Date.now();
      const last = this.lastMatch.get(d.top.personId);
      this.lastMatch.set(d.top.personId, now);
      if (last !== undefined && now - last < cfg.pipeline.cooldownMs) return; // suppressed repeat (spec §7.4)
    }
    await this.publish(ctx, { status, framesUsed: embs.length, decision: d }, best[0]);
  }

  private async publish(
    ctx: TrackCtx,
    r: { status: RecognitionEvent['status']; framesUsed: number; decision?: ReturnType<typeof identifyBurst>; reasons?: Record<string, number> },
    bestCand?: BurstCandidate,
  ): Promise<void> {
    const cfg = this.deps.cfg();
    const eventId = newId();
    const d = r.decision;
    const ev: RecognitionEvent = {
      type: 'recognition.result',
      eventId,
      ts: nowIso(),
      sourceId: this.id,
      trackId: ctx.track.id,
      status: r.status,
      score: r4(d?.score ?? null),
      secondScore: r4(d?.secondScore ?? null),
      frameAgreement: d ? r4(d.frameAgreement) : null,
      framesUsed: r.framesUsed,
      attempt: ctx.attempt,
      latencyMs: Date.now() - ctx.track.firstSeenWall,
      liveness: null,
    };
    if (r.status === 'match' && d?.top) {
      const p = this.deps.person(d.top.personId);
      if (p) ev.person = { id: p.id, externalId: p.externalId, firstName: p.firstName, lastName: p.lastName };
      ev.personId = d.top.personId;
    }
    if (r.reasons && Object.keys(r.reasons).length) ev.reasons = r.reasons;
    if (cfg.events.storeSnapshots && bestCand?.frame) {
      try {
        ev.snapshotId = newId();
        const jpeg = await encodeJpegBgr(bestCand.frame.data, bestCand.frame.width, bestCand.frame.height, 85);
        const file = this.deps.journal.snapshotPath(ev.snapshotId, ev.ts);
        await fs.mkdir(path.dirname(file), { recursive: true });
        await fs.writeFile(file, this.deps.cipher.encrypt(jpeg));
      } catch (e) {
        delete ev.snapshotId;
        this.deps.log.error({ err: (e as Error).message }, 'snapshot write failed');
      }
    }
    if (r.status !== 'low_quality' && r.status !== 'uncertain') this.deps.metrics.record(ev.latencyMs);
    this.deps.bus.publish(ev);
  }
}
