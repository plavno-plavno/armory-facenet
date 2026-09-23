// Gallery index lifecycle: load cache or rebuild from embeddings/, debounced cache flush (spec §8.3).

import { createHash } from 'node:crypto';
import type { Person } from '@faceid/shared';
import type { FileStore } from '../store/file-store.js';
import type { Logger } from '../util/logger.js';
import { GalleryIndex, type GalleryRow } from './index.js';

interface CacheMeta {
  version: 1;
  modelKey: string;
  dim: number;
  checksum: string;
  rows: GalleryRow[];
}

/** Checksum over the list of embeddings the index must contain for `modelKey`. */
export function expectedRows(persons: Iterable<Person>, modelKey: string): GalleryRow[] {
  const rows: GalleryRow[] = [];
  for (const p of persons) for (const ph of p.photos) if (ph.modelKeys.includes(modelKey)) rows.push({ personId: p.id, photoId: ph.id });
  rows.sort((a, b) => (a.personId + a.photoId < b.personId + b.photoId ? -1 : 1));
  return rows;
}

export function rowsChecksum(rows: GalleryRow[]): string {
  const h = createHash('sha256');
  for (const r of rows) h.update(`${r.personId}/${r.photoId}\n`);
  return h.digest('hex');
}

export class IndexManager {
  private timer: NodeJS.Timeout | null = null;
  private flushing: Promise<void> | null = null;

  constructor(
    readonly index: GalleryIndex,
    readonly modelKey: string,
    private readonly store: FileStore,
    private readonly log: Logger,
    private readonly debounceMs = 2000,
  ) {}

  /** Returns how the index was obtained. */
  async load(persons: Person[]): Promise<{ source: 'cache' | 'rebuild'; missing: GalleryRow[] }> {
    const rows = expectedRows(persons, this.modelKey);
    const checksum = rowsChecksum(rows);
    try {
      const cache = await this.store.readIndexCache(this.modelKey);
      const meta = cache?.meta as CacheMeta | undefined;
      if (cache && meta && meta.version === 1 && meta.checksum === checksum && meta.dim === this.index.dim) {
        this.index.load(cache.matrix, meta.rows);
        this.applyStatus(persons);
        return { source: 'cache', missing: [] };
      }
      if (cache) this.log.warn('gallery cache is stale, rebuilding');
    } catch (e) {
      this.log.warn({ err: (e as Error).message }, 'gallery cache is unreadable, rebuilding');
    }
    const missing: GalleryRow[] = [];
    this.index.clear();
    for (const r of rows) {
      try {
        this.index.upsertPhoto(r.personId, r.photoId, await this.store.readEmbedding(r.personId, r.photoId, this.modelKey));
      } catch (e) {
        missing.push(r);
        this.log.error({ personId: r.personId, photoId: r.photoId, err: (e as Error).message }, 'embedding unreadable');
      }
    }
    this.applyStatus(persons);
    await this.flushNow();
    return { source: 'rebuild', missing };
  }

  private applyStatus(persons: Person[]): void {
    for (const p of persons) this.index.setPersonActive(p.id, p.status !== 'disabled');
  }

  scheduleFlush(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = setTimeout(() => {
      this.timer = null;
      void this.flushNow().catch((e) => this.log.error({ err: (e as Error).message }, 'gallery cache flush failed'));
    }, this.debounceMs);
    this.timer.unref?.();
  }

  async flushNow(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    while (this.flushing) await this.flushing;
    const snap = this.index.snapshot();
    // Store rows in checksum order so cache validation is independent of insertion order.
    const order = snap.rows.map((r, i) => ({ r, i })).sort((a, b) => (a.r.personId + a.r.photoId < b.r.personId + b.r.photoId ? -1 : 1));
    const dim = this.index.dim;
    const matrix = new Float32Array(snap.matrix.length);
    order.forEach(({ i }, j) => matrix.set(snap.matrix.subarray(i * dim, (i + 1) * dim), j * dim));
    const rows = order.map((o) => o.r);
    const meta: CacheMeta = { version: 1, modelKey: this.modelKey, dim, checksum: rowsChecksum(rows), rows };
    this.flushing = this.store.writeIndexCache(this.modelKey, matrix, meta);
    try {
      await this.flushing;
    } finally {
      this.flushing = null;
    }
  }
}
