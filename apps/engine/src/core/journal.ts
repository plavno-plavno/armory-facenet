// Recognition journal (spec §10.5, §12.4): logs/recognitions/YYYY-MM-DD.jsonl, identifiers + scores only.

import { existsSync, promises as fs } from 'node:fs';
import path from 'node:path';
import type { RecognitionEvent } from '@faceid/shared';

export interface JournalEntry {
  eventId: string;
  ts: string;
  sourceId: string;
  trackId: string;
  status: RecognitionEvent['status'];
  personId?: string;
  score: number | null;
  secondScore: number | null;
  frameAgreement: number | null;
  framesUsed: number;
  attempt: number;
  latencyMs: number;
  snapshotId?: string;
}

export interface JournalQuery {
  from?: string;
  to?: string;
  personId?: string;
  sourceId?: string;
  status?: string;
  limit: number;
  cursor?: string; // eventId: return entries strictly older than this
}

export class RecognitionJournal {
  readonly dir: string;
  readonly snapshotsDir: string;
  private chain: Promise<void> = Promise.resolve();

  constructor(dataDir: string) {
    this.dir = path.join(dataDir, 'logs', 'recognitions');
    this.snapshotsDir = path.join(dataDir, 'snapshots');
  }

  static entryOf(e: RecognitionEvent): JournalEntry {
    return {
      eventId: e.eventId,
      ts: e.ts,
      sourceId: e.sourceId,
      trackId: e.trackId,
      status: e.status,
      personId: e.person?.id ?? e.personId,
      score: e.score,
      secondScore: e.secondScore,
      frameAgreement: e.frameAgreement,
      framesUsed: e.framesUsed,
      attempt: e.attempt,
      latencyMs: e.latencyMs,
      snapshotId: e.snapshotId,
    };
  }

  append(e: RecognitionEvent): Promise<void> {
    const entry = RecognitionJournal.entryOf(e);
    const file = path.join(this.dir, `${entry.ts.slice(0, 10)}.jsonl`);
    this.chain = this.chain
      .then(async () => {
        await fs.mkdir(this.dir, { recursive: true });
        await fs.appendFile(file, JSON.stringify(entry) + '\n');
      })
      .catch(() => undefined);
    return this.chain;
  }

  private async days(): Promise<string[]> {
    if (!existsSync(this.dir)) return [];
    return (await fs.readdir(this.dir)).filter((f) => f.endsWith('.jsonl')).map((f) => f.slice(0, 10)).sort().reverse();
  }

  private async readDay(day: string): Promise<JournalEntry[]> {
    const raw = await fs.readFile(path.join(this.dir, `${day}.jsonl`), 'utf8');
    const out: JournalEntry[] = [];
    for (const line of raw.split('\n')) {
      if (!line) continue;
      try {
        out.push(JSON.parse(line));
      } catch {
        /* torn last line after a crash */
      }
    }
    return out;
  }

  /** Newest first; cursor = last eventId of previous page (UUIDv7 is time-ordered). */
  async query(q: JournalQuery): Promise<{ items: JournalEntry[]; nextCursor: string | null }> {
    await this.chain;
    const items: JournalEntry[] = [];
    for (const day of await this.days()) {
      if (q.from && day < q.from.slice(0, 10)) break;
      if (q.to && day > q.to.slice(0, 10)) continue;
      const entries = (await this.readDay(day)).sort((a, b) => (a.eventId < b.eventId ? 1 : -1));
      for (const e of entries) {
        if (q.cursor && e.eventId >= q.cursor) continue;
        if (q.from && e.ts < q.from) continue;
        if (q.to && e.ts > q.to) continue;
        if (q.personId && e.personId !== q.personId) continue;
        if (q.sourceId && e.sourceId !== q.sourceId) continue;
        if (q.status && e.status !== q.status) continue;
        items.push(e);
        if (items.length > q.limit) break;
      }
      if (items.length > q.limit) break;
    }
    const more = items.length > q.limit;
    const page = items.slice(0, q.limit);
    return { items: page, nextCursor: more ? page[page.length - 1].eventId : null };
  }

  /** Remove every journal line and snapshot that refers to `personId` (DELETE ?purgeEvents=true). */
  async purgePerson(personId: string): Promise<number> {
    await this.chain;
    let removed = 0;
    for (const day of await this.days()) {
      const entries = await this.readDay(day);
      const keep = entries.filter((e) => e.personId !== personId);
      const drop = entries.filter((e) => e.personId === personId);
      if (drop.length === 0) continue;
      removed += drop.length;
      for (const e of drop) if (e.snapshotId) await fs.rm(this.snapshotPath(e.snapshotId, e.ts), { force: true });
      const file = path.join(this.dir, `${day}.jsonl`);
      await fs.writeFile(`${file}.tmp`, keep.map((e) => JSON.stringify(e) + '\n').join(''));
      await fs.rename(`${file}.tmp`, file);
    }
    return removed;
  }

  snapshotPath(snapshotId: string, ts: string): string {
    if (!/^[0-9a-f-]{36}$/.test(snapshotId)) throw new Error('Invalid snapshot id');
    return path.join(this.snapshotsDir, ts.slice(0, 10), `${snapshotId}.jpg.enc`);
  }

  async findSnapshot(snapshotId: string): Promise<string | null> {
    if (!/^[0-9a-f-]{36}$/.test(snapshotId) || !existsSync(this.snapshotsDir)) return null;
    for (const day of await fs.readdir(this.snapshotsDir)) {
      const p = path.join(this.snapshotsDir, day, `${snapshotId}.jpg.enc`);
      if (existsSync(p)) return p;
    }
    return null;
  }

  /** Retention (spec §11 events.*): drop journal files and snapshot folders older than N days. */
  async applyRetention(logDays: number, snapshotDays: number, now = Date.now()): Promise<void> {
    const cutoff = (days: number) => new Date(now - days * 86400_000).toISOString().slice(0, 10);
    const logCut = cutoff(logDays);
    for (const day of await this.days()) if (day < logCut) await fs.rm(path.join(this.dir, `${day}.jsonl`), { force: true });
    if (existsSync(this.snapshotsDir)) {
      const snapCut = cutoff(snapshotDays);
      for (const day of await fs.readdir(this.snapshotsDir)) {
        if (day < snapCut) await fs.rm(path.join(this.snapshotsDir, day), { recursive: true, force: true });
      }
    }
  }
}
