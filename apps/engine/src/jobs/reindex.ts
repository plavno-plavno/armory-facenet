// Background jobs; reindex recomputes embeddings of all photos with the current model (spec §8.5).

import type { Job } from '@faceid/shared';
import type { FaceAnalyzer } from '../core/analyzer.js';
import type { EventBus } from '../core/events.js';
import type { PersonService } from '../core/persons.js';
import type { FileStore } from '../store/file-store.js';
import { newId, nowIso } from '../util/ids.js';
import type { Logger } from '../util/logger.js';
import type { VisionStack } from '../vision/index.js';
import { decodeImage } from '../vision/image.js';
import { rectIoU } from '../vision/yunet.js';

export class JobManager {
  private readonly jobs = new Map<string, Job>();

  get(id: string): Job | undefined {
    return this.jobs.get(id);
  }

  running(kind?: string): Job | undefined {
    return [...this.jobs.values()].find((j) => j.state === 'running' && (!kind || j.kind === kind));
  }

  start(kind: string, total: number): Job {
    const job: Job = { id: newId(), kind, state: 'running', done: 0, total, errors: [], startedAt: nowIso(), finishedAt: null };
    this.jobs.set(job.id, job);
    return job;
  }
}

export interface ReindexOptions {
  /** Only photos that do not have an embedding for the current model yet. */
  onlyMissing: boolean;
}

export async function runReindex(
  job: Job,
  deps: { persons: PersonService; store: FileStore; vision: VisionStack; analyzer: FaceAnalyzer; bus: EventBus; log: Logger },
  opts: ReindexOptions,
): Promise<void> {
  const { persons, store, vision, analyzer, bus, log } = deps;
  const modelKey = vision.embedder.modelKey;
  const work = persons
    .all()
    .flatMap((p) => p.photos.filter((ph) => !opts.onlyMissing || !ph.modelKeys.includes(modelKey)).map((ph) => ({ p, ph })));
  job.total = work.length;
  const progress = () =>
    bus.publish({ type: 'index.progress', ts: nowIso(), jobId: job.id, done: job.done, total: job.total, state: job.state });
  progress();
  let lastEmit = 0;
  for (const { p, ph } of work) {
    try {
      if (!persons.tryGet(p.id)) continue; // deleted meanwhile
      const { frame } = await decodeImage(await store.readOriginal(p.id, ph.id));
      const dets = await analyzer.detectAll(frame);
      // The stored faceBox identifies which face was enrolled.
      const det = dets.reduce<{ d: (typeof dets)[number] | null; v: number }>(
        (best, d) => {
          const v = rectIoU(d.box, ph.faceBox);
          return v > best.v ? { d, v } : best;
        },
        { d: null, v: 0 },
      ).d;
      if (!det) throw new Error('enrolled face not found in original');
      const [emb] = await vision.embedder.embed([vision.aligner.align(frame, det.landmarks)]);
      await persons.markPhotoModelKey(p.id, ph.id, modelKey, emb);
    } catch (e) {
      job.errors.push({ personId: p.id, photoId: ph.id, message: (e as Error).message });
      log.error({ personId: p.id, photoId: ph.id, err: (e as Error).message }, 'reindex failed for photo');
    }
    job.done++;
    if (Date.now() - lastEmit > 500) {
      lastEmit = Date.now();
      progress();
    }
  }
  job.state = 'completed';
  job.finishedAt = nowIso();
  progress();
}
