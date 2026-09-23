import type { EngineEvent, RecognitionEvent } from '@faceid/shared';
import pino from 'pino';
import { describe, expect, it } from 'vitest';
import { EngineConfig } from '../../src/config/schema.js';
import { EventBus } from '../../src/core/events.js';
import { RecognitionJournal } from '../../src/core/journal.js';
import { GalleryIndex } from '../../src/gallery/index.js';
import { Burst } from '../../src/pipeline/burst.js';
import { decide, identifyBurst } from '../../src/pipeline/decision.js';
import { BaseSource } from '../../src/pipeline/sources/types.js';
import { PipelineMetrics, StreamRuntime } from '../../src/pipeline/stream-runtime.js';
import { IouTracker } from '../../src/pipeline/tracker.js';
import { PlainCipher } from '../../src/store/crypto.js';
import { l2normalize } from '../../src/vision/embedder.js';
import type { AlignedFace, Detection, Frame, LivenessChecker } from '../../src/vision/types.js';
import { NoopLivenessChecker } from '../../src/vision/types.js';

const D = 8;
const unit = (i: number) => {
  const v = new Float32Array(D);
  v[i] = 1;
  return v;
};
const mix = (a: Float32Array, b: Float32Array, wa: number) => l2normalize(a.map((x, i) => x * wa + b[i] * (1 - wa)));

describe('decision (§7.7)', () => {
  const m = EngineConfig.parse({}).match; // accept .45 reject .30 margin .05 agreement .6
  it('match / unknown / uncertain', () => {
    expect(decide(0.6, 0.2, 1, m)).toBe('match');
    expect(decide(0.6, 0.58, 1, m)).toBe('uncertain'); // margin
    expect(decide(0.6, 0.2, 0.4, m)).toBe('uncertain'); // agreement
    expect(decide(0.4, 0.1, 1, m)).toBe('uncertain'); // between thresholds
    expect(decide(0.29, 0.1, 1, m)).toBe('unknown');
    expect(decide(null, null, 0, m)).toBe('unknown');
    expect(decide(0.5, null, 1, m)).toBe('match');
  });

  it('burst probe is the normalized mean; agreement counts per-frame top-1', () => {
    const g = new GalleryIndex(D);
    g.upsertPhoto('A', 'a1', unit(0));
    g.upsertPhoto('B', 'b1', unit(1));
    const frames = [mix(unit(0), unit(1), 0.9), mix(unit(0), unit(1), 0.8), mix(unit(0), unit(1), 0.3)];
    const d = identifyBurst(g, frames, m);
    expect(d.top?.personId).toBe('A');
    expect(d.frameAgreement).toBeCloseTo(2 / 3);
  });
});

describe('IoU tracker (§7.3)', () => {
  const det = (x: number): Detection => ({ box: [x, 0, 100, 100], score: 0.99, landmarks: [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]] });
  it('keeps ids across frames, creates new tracks, expires after ttl', () => {
    const t = new IouTracker(0.3, 1000);
    const a = t.update([det(0), det(500)], 0).updated;
    expect(a.map((x) => x.id)).toEqual(['t-000001', 't-000002']);
    const b = t.update([det(10)], 100);
    expect(b.updated[0].id).toBe('t-000001');
    const c = t.update([det(12)], 1050);
    expect(c.expired.map((x) => x.id)).toEqual(['t-000002']);
    expect(c.updated[0].id).toBe('t-000001');
    const d = t.update([det(900)], 1200);
    expect(d.updated[0].id).toBe('t-000003');
  });
});

describe('burst (§7.5)', () => {
  it('completes by frame count or window; selects top by qualityScore', () => {
    const cfg = { maxFrames: 3, windowMs: 1000, topK: 2, minGood: 2 };
    const b = new Burst(0);
    const q = (score: number, passed = true) => ({ quality: { passed, qualityScore: score, reasons: passed ? [] : ['yaw'] } }) as never;
    b.add(q(0.5));
    b.add(q(0.9, false));
    expect(b.isComplete(100, cfg)).toBe(false);
    expect(b.isComplete(1000, cfg)).toBe(true);
    b.add(q(0.7));
    expect(b.isComplete(200, cfg)).toBe(true);
    expect(b.enoughGood(cfg)).toBe(true);
    expect(b.best(2).map((c) => c.quality.qualityScore)).toEqual([0.7, 0.5]);
    expect(b.reasons).toEqual({ yaw: 1 });
  });
});

// ---------- StreamRuntime with scripted detector/embedder ----------

class ManualSource extends BaseSource {
  async start() {
    this.setStatus('running');
  }
  async stop() {
    this.setStatus('stopped');
  }
  push(f: Frame) {
    this.emitFrame(f);
  }
}

interface Scene {
  faces: { x: number; size?: number; who: Float32Array }[];
}

function makeRuntime(cfgPatch: Record<string, unknown> = {}, liveness: LivenessChecker = new NoopLivenessChecker()) {
  const cfg = EngineConfig.parse({
    quality: { minSharpness: 0, minFaceSize: 80, requireInFrame: false, brightness: [0, 255] },
    pipeline: { detectFps: 60, cooldownMs: 10000, maxRetries: 2, maxConcurrentTracks: 3 },
    burst: { maxFrames: 5, windowMs: 100000, topK: 3, minGood: 3 },
    ...cfgPatch,
  });
  const g = new GalleryIndex(D);
  g.upsertPhoto('A', 'a1', unit(0));
  g.upsertPhoto('B', 'b1', unit(1));
  let scene: Scene = { faces: [] };
  const whoByBox = new Map<string, Float32Array>();
  const detector = {
    async detect(): Promise<Detection[]> {
      return scene.faces.map((f) => {
        const s = f.size ?? 120;
        const d: Detection = {
          box: [f.x, 100, s, s],
          score: 0.99,
          landmarks: [[f.x + s * 0.3, 140], [f.x + s * 0.7, 140], [f.x + s * 0.5, 160], [f.x + s * 0.35, 190], [f.x + s * 0.65, 190]],
        };
        whoByBox.set(String(d.landmarks[0][0]), f.who);
        return d;
      });
    },
  };
  const aligner = {
    align(_f: Frame, lm: Detection['landmarks']): AlignedFace {
      const rgb = new Uint8Array(112 * 112 * 3).fill(128);
      return { rgb, size: 112, transform: [lm[0][0]] };
    },
  };
  let embedCalls = 0;
  const embedder = {
    modelKey: 'fake@1',
    dim: D,
    async embed(faces: AlignedFace[]) {
      embedCalls++;
      return faces.map((f) => {
        return whoByBox.get(String(f.transform[0]))!;
      });
    },
  };
  const events: RecognitionEvent[] = [];
  const bus = new EventBus();
  bus.subscribe((e: EngineEvent) => e.type === 'recognition.result' && events.push(e));
  const src = new ManualSource('s1');
  const rt = new StreamRuntime('s1', { name: 's1', type: 'file', enabled: true }, src, {
    vision: { detector, aligner, embedder } as never,
    cfg: () => cfg,
    index: () => g,
    person: (id) => ({ id, firstName: id, lastName: id }),
    bus,
    journal: new RecognitionJournal('/nonexistent'),
    cipher: new PlainCipher(),
    liveness,
    metrics: new PipelineMetrics(),
    log: pino({ level: 'silent' }),
  });
  let ts = 0;
  const step = async (n: number, dt = 100) => {
    for (let i = 0; i < n; i++) {
      ts += dt;
      await rt.processFrame({ data: new Uint8Array(3), width: 1920, height: 1080, ts, sourceId: 's1' });
      await new Promise((r) => setImmediate(r));
    }
    await new Promise((r) => setTimeout(r, 5));
  };
  return { rt, src, events, step, setScene: (s: Scene) => (scene = s), embedCalls: () => embedCalls };
}

describe('track state machine (§7.4)', () => {
  it('NEW -> COLLECTING -> IDENTIFYING -> match, then no repeat while the track lives', async () => {
    const h = makeRuntime();
    h.setScene({ faces: [{ x: 100, who: unit(0) }] });
    await h.step(5);
    expect(h.events.map((e) => [e.status, e.person?.id, e.attempt])).toEqual([['match', 'A', 1]]);
    expect(h.events[0].framesUsed).toBe(3);
    await h.step(20);
    expect(h.events).toHaveLength(1);
  });

  it('cooldown suppresses a repeated match of the same person on a new track', async () => {
    const h = makeRuntime({ tracker: { ttlMs: 300 } });
    h.setScene({ faces: [{ x: 100, who: unit(0) }] });
    await h.step(5);
    h.setScene({ faces: [] });
    await h.step(5); // track expires
    h.setScene({ faces: [{ x: 100, who: unit(0) }] });
    await h.step(6);
    expect(h.events.filter((e) => e.status === 'match')).toHaveLength(1);
  });

  it('uncertain is retried up to maxRetries, then unknown', async () => {
    const h = makeRuntime();
    // between reject (0.30) and accept (0.45): cosine 0.4 to A, ~0.1 to B
    const v = l2normalize(Float32Array.from([0.4, 0.1, 0.9, 0, 0, 0, 0, 0]));
    h.setScene({ faces: [{ x: 100, who: v }] });
    await h.step(20);
    expect(h.events.map((e) => `${e.status}#${e.attempt}`)).toEqual(['uncertain#1', 'uncertain#2', 'unknown#3']);
    expect(h.events.every((e) => e.person === undefined)).toBe(true);
  });

  it('stranger -> unknown right away', async () => {
    const h = makeRuntime();
    h.setScene({ faces: [{ x: 100, who: unit(5) }] });
    await h.step(6);
    expect(h.events.map((e) => e.status)).toEqual(['unknown']);
  });

  it('low_quality bursts publish reasons and stop after maxRetries', async () => {
    const h = makeRuntime();
    h.setScene({ faces: [{ x: 100, size: 40, who: unit(0) }] });
    await h.step(30);
    expect(h.events.map((e) => `${e.status}#${e.attempt}`)).toEqual(['low_quality#1', 'low_quality#2', 'low_quality#3']);
    expect(h.events[0].reasons?.faceSize).toBe(5);
    expect(h.embedCalls()).toBe(0);
  });

  it('only maxConcurrentTracks largest faces are recognized', async () => {
    const h = makeRuntime({ pipeline: { maxConcurrentTracks: 2, detectFps: 60 } });
    h.setScene({
      faces: [
        { x: 0, size: 100, who: unit(5) },
        { x: 300, size: 200, who: unit(0) },
        { x: 700, size: 150, who: unit(1) },
      ],
    });
    await h.step(5);
    const ids = h.events.map((e) => e.person?.id).sort();
    expect(ids).toEqual(['A', 'B']);
  });
});

/** Fixed-verdict liveness checker; counts the frames it was asked about. */
function fakeLiveness(score: number, enabled = true) {
  const seen = { samples: 0, checks: 0 };
  const checker: LivenessChecker = {
    enabled: () => enabled,
    sample: () => ({ crops: [] }),
    async check(samples) {
      seen.samples += samples.length;
      seen.checks++;
      return { live: score >= 0.5, score };
    },
  };
  return { checker, seen };
}

describe('liveness (§7.9)', () => {
  it('spoof is final: no identification, nobody named, no retries', async () => {
    const l = fakeLiveness(0.1);
    const h = makeRuntime({}, l.checker);
    h.setScene({ faces: [{ x: 100, who: unit(0) }] });
    await h.step(20);
    expect(h.events.map((e) => e.status)).toEqual(['spoof']);
    expect(h.events[0].person).toBeUndefined();
    expect(h.events[0].score).toBeNull();
    expect(h.events[0].liveness).toEqual({ live: false, score: 0.1 });
    expect(h.embedCalls()).toBe(0);
    expect(l.seen.samples).toBe(3); // topK best frames
  });

  it('live face is identified as usual and the liveness result is published', async () => {
    const l = fakeLiveness(0.97);
    const h = makeRuntime({}, l.checker);
    h.setScene({ faces: [{ x: 100, who: unit(0) }] });
    await h.step(5);
    expect(h.events.map((e) => e.status)).toEqual(['match']);
    expect(h.events[0].person?.id).toBe('A');
    expect(h.events[0].liveness).toEqual({ live: true, score: 0.97 });
  });

  it('disabled checker is never called and liveness stays null', async () => {
    const l = fakeLiveness(0.1, false);
    const h = makeRuntime({}, l.checker);
    h.setScene({ faces: [{ x: 100, who: unit(0) }] });
    await h.step(5);
    expect(h.events.map((e) => e.status)).toEqual(['match']);
    expect(h.events[0].liveness).toBeNull();
    expect(l.seen.checks).toBe(0);
  });
});

describe('backpressure (§7.2)', () => {
  it('keeps only the newest frame and counts dropped frames', async () => {
    const h = makeRuntime({ pipeline: { detectFps: 60 } });
    h.setScene({ faces: [] });
    await h.rt.start();
    for (let i = 0; i < 50; i++) h.src.push({ data: new Uint8Array(3), width: 10, height: 10, ts: i, sourceId: 's1' });
    await new Promise((r) => setTimeout(r, 50));
    expect(h.rt.received).toBe(50);
    expect(h.rt.processed).toBeLessThan(50);
    expect(h.rt.processed + h.rt.dropped).toBe(50);
  });
});
