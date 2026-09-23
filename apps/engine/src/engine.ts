// Engine composition root: wires config, storage, vision, gallery, pipeline and services.

import { mkdirSync } from 'node:fs';
import { ConfigStore } from './config/config-store.js';
import type { EngineConfig } from './config/schema.js';
import { FaceAnalyzer } from './core/analyzer.js';
import { EventBus } from './core/events.js';
import { RecognitionJournal } from './core/journal.js';
import { PersonService } from './core/persons.js';
import { RecognizeService } from './core/recognize.js';
import { WebhookDispatcher } from './core/webhooks.js';
import { GalleryIndex } from './gallery/index.js';
import { IndexManager } from './gallery/rebuild.js';
import { JobManager, runReindex } from './jobs/reindex.js';
import { releaseStaleCaptures } from './pipeline/sources/device-lock.js';
import { FfmpegWebcamProvider } from './pipeline/sources/webcam.js';
import type { WebcamProvider } from './pipeline/sources/types.js';
import { StreamManager } from './pipeline/stream-manager.js';
import { PipelineMetrics } from './pipeline/stream-runtime.js';
import { TokenStore } from './server/auth.js';
import { AuditLog } from './store/audit.js';
import { AesGcmCipher, PlainCipher, type Cipher } from './store/crypto.js';
import { FileStore } from './store/file-store.js';
import { createLogger, type Logger } from './util/logger.js';
import { prepareCuda } from './vision/cuda.js';
import { loadVision, OnnxLivenessChecker, type VisionStack } from './vision/index.js';

export interface EngineOptions {
  dataDir: string;
  modelsDir: string;
  /** Unwrapped data-encryption key; required unless privacy.encryptAtRest = false. */
  dek?: Buffer;
  configOverrides?: Record<string, unknown>;
  webcams?: WebcamProvider;
  logger?: Logger;
  verifyModelChecksums?: boolean;
}

export class Engine {
  readonly startedAt = Date.now();
  readonly warnings: string[] = [];
  readonly jobs = new JobManager();

  private constructor(
    readonly opts: EngineOptions,
    readonly bus: EventBus,
    readonly metrics: PipelineMetrics,
    readonly cfgStore: ConfigStore,
    readonly log: Logger,
    readonly cipher: Cipher,
    readonly vision: VisionStack,
    readonly store: FileStore,
    readonly audit: AuditLog,
    readonly journal: RecognitionJournal,
    readonly indexMgr: IndexManager,
    readonly analyzer: FaceAnalyzer,
    readonly persons: PersonService,
    readonly recognize: RecognizeService,
    readonly streams: StreamManager,
    readonly tokens: TokenStore,
    readonly webhooks: WebhookDispatcher,
  ) {}

  cfg(): EngineConfig {
    return this.cfgStore.get();
  }

  get index(): GalleryIndex {
    return this.indexMgr.index;
  }

  static async create(opts: EngineOptions): Promise<Engine> {
    mkdirSync(opts.dataDir, { recursive: true });
    const cfgStore = new ConfigStore(opts.dataDir, opts.configOverrides);
    const cfg = () => cfgStore.get();
    const log = opts.logger ?? createLogger({ level: cfg().log.level, dataDir: opts.dataDir });

    let cipher: Cipher;
    const warnings: string[] = [];
    if (cfg().privacy.encryptAtRest) {
      if (!opts.dek) throw new Error('Encryption at rest is enabled but no data key was provided');
      cipher = new AesGcmCipher(opts.dek);
    } else {
      cipher = new PlainCipher();
      warnings.push('privacy.encryptAtRest=false: biometric data is stored unencrypted (development only)');
      log.warn('privacy.encryptAtRest=false: biometric data is stored UNENCRYPTED - development only');
    }

    const vision = await loadVision({
      modelsDir: opts.modelsDir,
      embedderId: cfg().models.embedder,
      executionProvider: cfg().models.executionProvider,
      intraOpThreads: cfg().models.intraOpThreads,
      detector: cfg().detector,
      verifyChecksums: opts.verifyModelChecksums ?? true,
    });
    log.info({ detector: vision.info.detector, embedder: vision.info.embedder, liveness: vision.info.liveness }, 'models loaded');
    if (!vision.liveness.length && cfg().liveness.enabled) {
      const why = vision.info.liveness.missing.length ? `missing ${vision.info.liveness.missing.join(', ')} — run models/download.sh` : 'no liveness models in manifest.json';
      warnings.push(`Anti-spoofing is off (${why}): photos and screens are not rejected.`);
      log.warn({ missing: vision.info.liveness.missing }, 'liveness models unavailable, anti-spoofing disabled');
    }
    if (vision.info.embedder.fallbackReason) {
      const cuda = prepareCuda();
      const hint = cuda.missing.length ? ` Missing CUDA libraries: ${cuda.missing.join(', ')} — run "npm run cuda:install" or install the CUDA 13 runtime + cuDNN 9.` : '';
      warnings.push(`GPU not used, embeddings run on CPU.${hint}`);
      log.warn({ reason: vision.info.embedder.fallbackReason.slice(0, 500), missing: cuda.missing }, 'GPU execution provider unavailable, using CPU');
    }

    const store = new FileStore(opts.dataDir, cipher);
    const audit = new AuditLog(opts.dataDir);
    const journal = new RecognitionJournal(opts.dataDir);
    const index = new GalleryIndex(vision.embedder.dim);
    const indexMgr = new IndexManager(index, vision.embedder.modelKey, store, log);
    const analyzer = new FaceAnalyzer(vision, cfg);
    const bus = new EventBus();
    const persons = new PersonService(store, indexMgr, analyzer, audit, bus, journal, cfg, log);
    const recognize = new RecognizeService(analyzer, () => indexMgr.index, cfg);
    const tokens = new TokenStore(opts.dataDir, cipher);
    const webhooks = new WebhookDispatcher(cfg, log);
    const metrics = new PipelineMetrics();
    const streams = new StreamManager(
      opts.dataDir,
      {
        vision,
        cfg,
        index: () => indexMgr.index,
        person: (id) => persons.tryGet(id),
        bus,
        journal,
        cipher,
        liveness: new OnnxLivenessChecker(vision.liveness, () => cfg().liveness),
        metrics,
        log,
      },
      opts.webcams ?? new FfmpegWebcamProvider((m) => log.warn(m)),
      analyzer,
    );
    cfgStore.on('change', (c: EngineConfig) => vision.detector.setOptions(c.detector));
    const engine = new Engine(opts, bus, metrics, cfgStore, log, cipher, vision, store, audit, journal, indexMgr, analyzer, persons, recognize, streams, tokens, webhooks);
    engine.warnings.push(...warnings);
    return engine;
  }

  private retentionTimer: NodeJS.Timeout | null = null;

  async start(): Promise<void> {
    await this.tokens.init();
    const r = await this.persons.init();
    this.log.info(r, 'gallery ready');
    this.bus.subscribe((e) => {
      if (e.type === 'recognition.result') void this.journal.append(e);
      this.webhooks.handle(e);
    });
    // Free cameras still held by capture processes of an earlier (crashed / suspended) run.
    if (!this.opts.webcams) await releaseStaleCaptures((m) => this.log.warn(m));
    await this.streams.init();
    const missing = this.persons.all().reduce((n, p) => n + p.photos.filter((ph) => !ph.modelKeys.includes(this.indexMgr.modelKey)).length, 0);
    if (missing > 0) {
      this.log.warn({ missing }, 'photos without embeddings for the current model; starting reindex');
      this.startReindex(true);
    }
    const retention = () => {
      const e = this.cfg().events;
      void this.journal.applyRetention(e.logRetentionDays, e.snapshotRetentionDays).catch((err) => this.log.error({ err: err.message }, 'retention failed'));
    };
    retention();
    this.retentionTimer = setInterval(retention, 3600_000);
    this.retentionTimer.unref();
  }

  startReindex(onlyMissing: boolean): string {
    const running = this.jobs.running('reindex');
    if (running) return running.id;
    const job = this.jobs.start('reindex', 0);
    void runReindex(job, { persons: this.persons, store: this.store, vision: this.vision, analyzer: this.analyzer, bus: this.bus, log: this.log }, { onlyMissing }).catch((e) => {
      job.state = 'failed';
      job.finishedAt = new Date().toISOString();
      this.log.error({ err: (e as Error).message }, 'reindex job failed');
    });
    return job.id;
  }

  health() {
    const reindexing = !!this.jobs.running('reindex');
    const streams = this.streams.list();
    const anyStreamError = streams.some((s) => s.enabled && (s.status === 'error' || s.status === 'reconnecting'));
    const status = reindexing || anyStreamError || this.warnings.length ? 'degraded' : 'ok';
    const r1 = (v: number | null) => (v === null ? null : Math.round(v));
    return {
      status,
      warnings: this.warnings,
      uptimeSec: Math.round((Date.now() - this.startedAt) / 1000),
      models: {
        detector: this.vision.info.detector,
        embedder: this.vision.info.embedder,
        liveness: { ...this.vision.info.liveness, enabled: this.cfg().liveness.enabled && this.vision.liveness.length > 0 },
      },
      gallery: this.index.stats(),
      streams: streams.map((s) => ({ id: s.id, name: s.name, status: s.status, fps: s.fps, droppedFrames: s.droppedFrames })),
      metrics: {
        detectFps: Math.round(streams.reduce((a, s) => a + s.fps, 0) * 10) / 10,
        latencyP50: r1(this.metrics.percentile(50)),
        latencyP95: r1(this.metrics.percentile(95)),
        droppedFrames: streams.reduce((a, s) => a + s.droppedFrames, 0),
      },
      jobs: reindexing ? [this.jobs.running('reindex')] : [],
      restartRequired: this.cfgStore.pendingRestart(),
    };
  }

  async close(): Promise<void> {
    if (this.retentionTimer) clearInterval(this.retentionTimer);
    await this.streams.stopAll();
    await this.indexMgr.flushNow();
  }
}
