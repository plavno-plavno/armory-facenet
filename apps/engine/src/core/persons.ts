// Persons & photos: CRUD, enrollment rules (spec §9.2), write ordering (spec §8.4).

import type { Person, PersonCreate, PersonPatch, Photo } from '@faceid/shared';
import yazl from 'yazl';
import type { EngineConfig } from '../config/schema.js';
import type { GalleryIndex } from '../gallery/index.js';
import type { IndexManager } from '../gallery/rebuild.js';
import type { AuditLog } from '../store/audit.js';
import type { FileStore } from '../store/file-store.js';
import { KeyedMutex } from '../store/mutex.js';
import { ApiError } from '../util/errors.js';
import { newId, nowIso } from '../util/ids.js';
import type { Logger } from '../util/logger.js';
import type { Box } from '../vision/types.js';
import { dot } from '../vision/embedder.js';
import { publicQuality, type AnalyzedPhoto, type FaceAnalyzer } from './analyzer.js';
import type { EventBus } from './events.js';
import type { RecognitionJournal } from './journal.js';

export interface EnrollFlags {
  allowDuplicate?: boolean;
  force?: boolean;
  faceBox?: Box;
  /** Per-photo origin (e.g. frames captured by the admin UI from a stream); default "upload". */
  sources?: Photo['source'][];
}

export interface PhotoError {
  index: number;
  code: string;
  message: string;
  details: Record<string, unknown>;
}

const round3 = (v: number) => Math.round(v * 1000) / 1000;

export class PersonService {
  private readonly persons = new Map<string, Person>();
  private readonly byExternalId = new Map<string, string>();
  private readonly mutex = new KeyedMutex();

  constructor(
    private readonly store: FileStore,
    private readonly indexMgr: IndexManager,
    private readonly analyzer: FaceAnalyzer,
    private readonly audit: AuditLog,
    private readonly bus: EventBus,
    private readonly journal: RecognitionJournal,
    private readonly cfg: () => EngineConfig,
    private readonly log: Logger,
  ) {}

  get index(): GalleryIndex {
    return this.indexMgr.index;
  }

  get modelKey(): string {
    return this.indexMgr.modelKey;
  }

  async init(): Promise<{ persons: number; index: 'cache' | 'rebuild'; missing: number }> {
    const gc = await this.store.init();
    if (gc.removedDirs.length) this.log.warn({ count: gc.removedDirs.length }, 'removed incomplete person directories');
    const ids = await this.store.listPersonIds();
    const loaded: Person[] = [];
    const BATCH = 64;
    for (let i = 0; i < ids.length; i += BATCH) {
      const chunk = await Promise.all(
        ids.slice(i, i + BATCH).map(async (id) => {
          try {
            const p = await this.store.readPerson(id);
            await this.store.gcPerson(p);
            return p;
          } catch (e) {
            this.log.error({ personId: id, err: (e as Error).message }, 'cannot read person');
            return null;
          }
        }),
      );
      for (const p of chunk) if (p) loaded.push(p);
    }
    for (const p of loaded) this.remember(p);
    const res = await this.indexMgr.load(loaded);
    return { persons: loaded.length, index: res.source, missing: res.missing.length };
  }

  private remember(p: Person): void {
    const prev = this.persons.get(p.id);
    if (prev?.externalId) this.byExternalId.delete(prev.externalId);
    this.persons.set(p.id, p);
    if (p.externalId) this.byExternalId.set(p.externalId, p.id);
  }

  all(): Person[] {
    return [...this.persons.values()];
  }

  count(): number {
    return this.persons.size;
  }

  get(id: string): Person {
    const p = this.persons.get(id);
    if (!p) throw new ApiError('NOT_FOUND', 'Person not found');
    return p;
  }

  tryGet(id: string): Person | undefined {
    return this.persons.get(id);
  }

  list(q: { q?: string; status?: string; limit: number; cursor?: string }) {
    const needle = q.q?.trim().toLowerCase();
    let items = [...this.persons.values()];
    if (q.status) items = items.filter((p) => p.status === q.status);
    if (needle) {
      items = items.filter((p) =>
        [p.firstName, p.lastName, p.middleName, p.externalId, `${p.lastName} ${p.firstName}`, `${p.firstName} ${p.lastName}`]
          .filter(Boolean)
          .some((s) => s!.toLowerCase().includes(needle)),
      );
    }
    items.sort((a, b) => (a.id < b.id ? -1 : 1));
    const total = items.length;
    if (q.cursor) items = items.filter((p) => p.id > q.cursor!);
    const page = items.slice(0, q.limit);
    return { items: page, nextCursor: items.length > q.limit ? page[page.length - 1].id : null, total };
  }

  private checkExternalId(externalId: string | undefined | null, selfId?: string): void {
    if (!externalId) return;
    const owner = this.byExternalId.get(externalId);
    if (owner && owner !== selfId) throw new ApiError('VALIDATION_ERROR', 'externalId already in use', { field: 'externalId' }, 409);
  }

  private checkConsent(consent: PersonCreate['consent']): void {
    if (this.cfg().privacy.requireConsent && consent?.obtained !== true) {
      throw new ApiError('CONSENT_REQUIRED', 'Consent must be obtained before enrollment');
    }
  }

  /** Analyze photos; returns results or throws an aggregated error (nothing is written). */
  private async analyzeAll(bufs: Buffer[], faceBox?: Box): Promise<AnalyzedPhoto[]> {
    const results: AnalyzedPhoto[] = [];
    const errors: PhotoError[] = [];
    for (let i = 0; i < bufs.length; i++) {
      try {
        results.push(await this.analyzer.analyzeForEnrollment(bufs[i], i === 0 ? faceBox : undefined));
      } catch (e) {
        if (!(e instanceof ApiError)) throw e;
        errors.push({ index: i, code: e.code, message: e.message, details: e.details });
      }
    }
    if (errors.length) throw aggregate(errors);
    return results;
  }

  /** Duplicate check against other persons (spec §9.2). */
  private checkDuplicates(photos: AnalyzedPhoto[], selfId: string | undefined, flags: EnrollFlags): string[] {
    const accept = this.cfg().match.acceptThreshold;
    const errors: PhotoError[] = [];
    photos.forEach((ph, i) => {
      const [top] = this.index.match(ph.embedding, 1, { exclude: selfId, includeInactive: true });
      if (top && top.score >= accept) {
        const other = this.persons.get(top.personId);
        errors.push({
          index: i,
          code: 'DUPLICATE_SUSPECTED',
          message: 'Face matches another registered person',
          details: {
            candidate: {
              personId: top.personId,
              name: other ? `${other.lastName} ${other.firstName}` : undefined,
              score: round3(top.score),
            },
          },
        });
      }
    });
    if (errors.length === 0) return [];
    if (flags.allowDuplicate) return ['allowDuplicate'];
    throw aggregate(errors);
  }

  /** Photos must belong to the same person as existing references (or the first new photo). */
  private checkMismatch(photos: AnalyzedPhoto[], personId: string | undefined, flags: EnrollFlags): string[] {
    const reject = this.cfg().match.rejectThreshold;
    const errors: PhotoError[] = [];
    photos.forEach((ph, i) => {
      let score: number | null = null;
      if (personId) score = this.index.scorePerson(personId, ph.embedding);
      else if (i > 0) score = dot(ph.embedding, photos[0].embedding);
      if (score !== null && score < reject) {
        errors.push({ index: i, code: 'PHOTO_MISMATCH', message: 'Photo does not match the person', details: { score: round3(score) } });
      }
    });
    if (errors.length === 0) return [];
    if (flags.force) return ['force'];
    throw aggregate(errors);
  }

  private toPhoto(a: AnalyzedPhoto, id: string, source: Photo['source']): Photo {
    return {
      id,
      source,
      createdAt: nowIso(),
      faceBox: a.det.box.map((v) => Math.round(v)) as Box,
      quality: publicQuality(a.quality),
      modelKeys: [this.modelKey],
    };
  }

  private async writePhotos(
    personId: string,
    photos: AnalyzedPhoto[],
    source: Photo['source'] | Photo['source'][],
  ): Promise<{ meta: Photo; emb: Float32Array }[]> {
    const out: { meta: Photo; emb: Float32Array }[] = [];
    for (const [i, a] of photos.entries()) {
      const id = newId();
      await this.store.writePhotoFiles(personId, id, {
        originalJpeg: a.originalJpeg,
        alignedPng: a.alignedPng,
        embeddings: new Map([[this.modelKey, a.embedding]]),
      });
      out.push({ meta: this.toPhoto(a, id, Array.isArray(source) ? (source[i] ?? 'upload') : source), emb: a.embedding });
    }
    return out;
  }

  async create(data: PersonCreate, bufs: Buffer[], flags: EnrollFlags, actor: string): Promise<Person> {
    this.checkConsent(data.consent);
    this.checkExternalId(data.externalId);
    const max = this.cfg().enroll.maxPhotos;
    if (bufs.length < 1 || bufs.length > max) throw new ApiError('VALIDATION_ERROR', `Provide 1..${max} photos`);
    const analyzed = await this.analyzeAll(bufs, flags.faceBox);
    const bypass = [...this.checkMismatch(analyzed, undefined, flags), ...this.checkDuplicates(analyzed, undefined, flags)];
    const id = newId();
    return this.mutex.run(id, async () => {
      this.checkExternalId(data.externalId); // re-check: another create may have raced us
      // Order (spec §8.4): photo files -> embeddings -> person.json -> in-memory index.
      const written = await this.writePhotos(id, analyzed, flags.sources ?? 'upload');
      const now = nowIso();
      const person: Person = {
        id,
        ...data,
        consent: data.consent ? { ...data.consent, obtainedAt: data.consent.obtainedAt ?? (data.consent.obtained ? now : undefined) } : undefined,
        status: data.status ?? 'active',
        photos: written.map((w) => w.meta),
        createdAt: now,
        updatedAt: now,
        version: 1,
      };
      await this.store.writePerson(person);
      for (const w of written) this.index.upsertPhoto(id, w.meta.id, w.emb);
      this.index.setPersonActive(id, person.status !== 'disabled');
      this.indexMgr.scheduleFlush();
      this.remember(person);
      await this.audit.append({ actor, action: 'person.create', personId: id, flags: bypass, details: { photos: written.length } });
      this.bus.publish({ type: 'person.changed', ts: now, personId: id, action: 'created' });
      return person;
    });
  }

  async patch(id: string, patch: PersonPatch, ifMatch: number | undefined, actor: string): Promise<Person> {
    return this.mutex.run(id, async () => {
      const cur = this.get(id);
      if (ifMatch !== undefined && ifMatch !== cur.version) {
        throw new ApiError('VERSION_CONFLICT', 'Person was modified', { currentVersion: cur.version });
      }
      if (patch.externalId !== undefined) this.checkExternalId(patch.externalId, id);
      if (patch.consent && this.cfg().privacy.requireConsent && patch.consent.obtained !== true) {
        throw new ApiError('CONSENT_REQUIRED', 'Consent cannot be withdrawn by PATCH; delete the person instead');
      }
      const next: Record<string, unknown> = { ...cur };
      for (const [k, v] of Object.entries(patch)) {
        if (v === null) delete next[k];
        else next[k] = v;
      }
      const updated = { ...(next as Person), updatedAt: nowIso(), version: cur.version + 1 };
      await this.store.writePerson(updated);
      this.remember(updated);
      if (patch.status) this.index.setPersonActive(id, patch.status !== 'disabled');
      await this.audit.append({ actor, action: 'person.update', personId: id, details: { fields: Object.keys(patch).join(',') } });
      this.bus.publish({ type: 'person.changed', ts: updated.updatedAt, personId: id, action: 'updated' });
      return updated;
    });
  }

  async delete(id: string, purgeEvents: boolean, actor: string): Promise<void> {
    await this.mutex.run(id, async () => {
      const p = this.get(id);
      // Stop recognizing immediately, then remove files (spec §8.4, §15.4 item 5).
      this.index.removePerson(id);
      this.persons.delete(id);
      if (p.externalId) this.byExternalId.delete(p.externalId);
      await this.store.deletePerson(id);
      this.indexMgr.scheduleFlush();
      const purged = purgeEvents ? await this.journal.purgePerson(id) : 0;
      await this.audit.append({ actor, action: 'person.delete', personId: id, details: { purgeEvents, purged } });
      this.bus.publish({ type: 'person.changed', ts: nowIso(), personId: id, action: 'deleted' });
    });
  }

  async addPhotos(id: string, bufs: Buffer[], flags: EnrollFlags, actor: string): Promise<Photo[]> {
    this.get(id);
    const max = this.cfg().enroll.maxPhotos;
    if (bufs.length < 1 || bufs.length > max) throw new ApiError('VALIDATION_ERROR', `Provide 1..${max} photos`);
    const analyzed = await this.analyzeAll(bufs, flags.faceBox);
    return this.addAnalyzed(id, analyzed, flags.sources ?? 'upload', flags, actor);
  }

  /** Shared by upload and camera capture. */
  async addAnalyzed(id: string, analyzed: AnalyzedPhoto[], source: Photo['source'] | Photo['source'][], flags: EnrollFlags, actor: string): Promise<Photo[]> {
    return this.mutex.run(id, async () => {
      const cur = this.get(id);
      const bypass = [...this.checkMismatch(analyzed, id, flags), ...this.checkDuplicates(analyzed, id, flags)];
      const written = await this.writePhotos(id, analyzed, source);
      const updated: Person = { ...cur, photos: [...cur.photos, ...written.map((w) => w.meta)], updatedAt: nowIso(), version: cur.version + 1 };
      await this.store.writePerson(updated);
      for (const w of written) this.index.upsertPhoto(id, w.meta.id, w.emb);
      this.index.setPersonActive(id, updated.status !== 'disabled');
      this.indexMgr.scheduleFlush();
      this.remember(updated);
      await this.audit.append({ actor, action: 'photo.add', personId: id, flags: bypass, details: { photos: written.length, source: String(source) } });
      this.bus.publish({ type: 'person.changed', ts: updated.updatedAt, personId: id, action: 'photos_added' });
      return written.map((w) => w.meta);
    });
  }

  async deletePhoto(id: string, photoId: string, actor: string): Promise<void> {
    await this.mutex.run(id, async () => {
      const cur = this.get(id);
      if (!cur.photos.some((p) => p.id === photoId)) throw new ApiError('NOT_FOUND', 'Photo not found');
      if (cur.photos.length === 1) throw new ApiError('LAST_PHOTO', 'Cannot delete the last photo of a person');
      this.index.removePhoto(id, photoId);
      const updated: Person = { ...cur, photos: cur.photos.filter((p) => p.id !== photoId), updatedAt: nowIso(), version: cur.version + 1 };
      await this.store.writePerson(updated);
      await this.store.deletePhotoFiles(id, photoId);
      this.indexMgr.scheduleFlush();
      this.remember(updated);
      await this.audit.append({ actor, action: 'photo.delete', personId: id, photoId });
      this.bus.publish({ type: 'person.changed', ts: updated.updatedAt, personId: id, action: 'photo_deleted' });
    });
  }

  async photoImage(id: string, photoId: string, variant: 'original' | 'aligned'): Promise<{ data: Buffer; type: string }> {
    const p = this.get(id);
    if (!p.photos.some((ph) => ph.id === photoId)) throw new ApiError('NOT_FOUND', 'Photo not found');
    return variant === 'aligned'
      ? { data: await this.store.readAligned(id, photoId), type: 'image/png' }
      : { data: await this.store.readOriginal(id, photoId), type: 'image/jpeg' };
  }

  /** Subject access export (spec §10.3): person.json + original photos. */
  async exportZip(id: string, actor: string): Promise<Buffer> {
    const p = this.get(id);
    const zip = new yazl.ZipFile();
    zip.addBuffer(Buffer.from(JSON.stringify(p, null, 2)), 'person.json');
    for (const ph of p.photos) zip.addBuffer(await this.store.readOriginal(id, ph.id), `photos/${ph.id}.jpg`);
    zip.end();
    const chunks: Buffer[] = [];
    for await (const c of zip.outputStream) chunks.push(c as Buffer);
    await this.audit.append({ actor, action: 'person.export', personId: id });
    return Buffer.concat(chunks);
  }

  /** Mark that a photo now has an embedding for `modelKey` (reindex). */
  async markPhotoModelKey(personId: string, photoId: string, modelKey: string, emb: Float32Array): Promise<void> {
    await this.mutex.run(personId, async () => {
      const cur = this.persons.get(personId);
      if (!cur) return;
      const ph = cur.photos.find((p) => p.id === photoId);
      if (!ph) return;
      await this.store.writeEmbedding(personId, photoId, modelKey, emb);
      if (!ph.modelKeys.includes(modelKey)) {
        const updated: Person = {
          ...cur,
          photos: cur.photos.map((p) => (p.id === photoId ? { ...p, modelKeys: [...p.modelKeys, modelKey] } : p)),
        };
        await this.store.writePerson(updated);
        this.remember(updated);
      }
      if (modelKey === this.modelKey) {
        this.index.upsertPhoto(personId, photoId, emb);
        this.index.setPersonActive(personId, cur.status !== 'disabled');
        this.indexMgr.scheduleFlush();
      }
    });
  }
}

function aggregate(errors: PhotoError[]): ApiError {
  const first = errors[0];
  return new ApiError(first.code as ApiError['code'], first.message, { ...first.details, photoIndex: first.index, photos: errors });
}
