import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import pino from 'pino';
import { afterAll, describe, expect, it } from 'vitest';
import { ConfigStore, deepMerge } from '../../src/config/config-store.js';
import { RecognitionJournal } from '../../src/core/journal.js';
import { GalleryIndex } from '../../src/gallery/index.js';
import { IndexManager } from '../../src/gallery/rebuild.js';
import { writeFileAtomic } from '../../src/store/atomic.js';
import { AesGcmCipher, DecryptError } from '../../src/store/crypto.js';
import { decodeEmbedding, encodeEmbedding } from '../../src/store/emb-format.js';
import { FileStore } from '../../src/store/file-store.js';
import { loadOrCreateDek, passwordWrapper } from '../../src/store/keystore.js';
import { KeyedMutex } from '../../src/store/mutex.js';
import { l2normalize } from '../../src/vision/embedder.js';

const dirs: string[] = [];
const tmp = () => {
  const d = mkdtempSync(path.join(tmpdir(), 'faceid-unit-'));
  dirs.push(d);
  return d;
};
afterAll(() => dirs.forEach((d) => rmSync(d, { recursive: true, force: true })));

const rand = (dim: number) => l2normalize(Float32Array.from({ length: dim }, () => Math.random() - 0.5));

describe('crypto (§12.2)', () => {
  it('round-trips, uses a fresh nonce, detects tampering and wrong keys', () => {
    const c = new AesGcmCipher(randomBytes(32));
    const plain = Buffer.from('biometric data');
    const a = c.encrypt(plain);
    const b = c.encrypt(plain);
    expect(a.equals(b)).toBe(false);
    expect(c.decrypt(a).equals(plain)).toBe(true);
    const tampered = Buffer.from(a);
    tampered[tampered.length - 1] ^= 1;
    expect(() => c.decrypt(tampered)).toThrow(DecryptError);
    expect(() => new AesGcmCipher(randomBytes(32)).decrypt(a)).toThrow(DecryptError);
    expect(() => c.decrypt(plain)).toThrow(DecryptError);
  });

  it('password keystore: same DEK on reopen, wrong password fails', async () => {
    const d = tmp();
    const k1 = await loadOrCreateDek(d, passwordWrapper('pw1'));
    const k2 = await loadOrCreateDek(d, passwordWrapper('pw1'));
    expect(k1.equals(k2)).toBe(true);
    await expect(loadOrCreateDek(d, passwordWrapper('bad'))).rejects.toThrow();
    expect(readFileSync(path.join(d, 'keystore.bin')).includes(k1)).toBe(false);
  });
});

describe('embedding file format (§8.2)', () => {
  it('header layout and round trip', () => {
    const v = rand(512);
    const buf = encodeEmbedding(v);
    expect(buf.length).toBe(16 + 512 * 4);
    expect(buf.subarray(0, 4).toString()).toBe('FEMB');
    expect(buf.readUInt16LE(4)).toBe(1);
    expect(buf.readUInt16LE(6)).toBe(512);
    expect(decodeEmbedding(buf)).toEqual(v);
    expect(() => decodeEmbedding(buf.subarray(0, 100))).toThrow();
  });
});

describe('atomic write (§8.4)', () => {
  it('a process killed mid-write never leaves a torn file', async () => {
    const d = tmp();
    const target = path.join(d, 'file.bin');
    const A = Buffer.alloc(4 * 1024 * 1024, 0xaa);
    await writeFileAtomic(target, A);
    const script = `
      import { writeFileAtomic } from ${JSON.stringify(path.resolve(__dirname, '../../src/store/atomic.ts'))};
      const B = Buffer.alloc(4 * 1024 * 1024, 0xbb);
      const A = Buffer.alloc(4 * 1024 * 1024, 0xaa);
      for (let i = 0; ; i++) await writeFileAtomic(${JSON.stringify(target)}, i % 2 ? A : B);
    `;
    const scriptFile = path.join(d, 'writer.mts');
    writeFileSync(scriptFile, script);
    for (let round = 0; round < 5; round++) {
      const p = spawn(process.execPath, ['--import', 'tsx', scriptFile], { stdio: 'ignore', cwd: path.resolve(__dirname, '../../../..') });
      await new Promise((r) => setTimeout(r, 400 + round * 137));
      p.kill('SIGKILL');
      await new Promise((r) => p.once('close', r));
      const buf = readFileSync(target);
      expect(buf.length).toBe(A.length);
      expect(buf.every((x) => x === buf[0])).toBe(true);
    }
    // leftovers are only *.tmp files, which the store ignores/cleans
    expect(readdirSync(d).filter((f) => f !== 'file.bin' && f !== 'writer.mts').every((f) => f.endsWith('.tmp'))).toBe(true);
  }, 60000);
});

describe('gallery index (§8.3, §8.6)', () => {
  it('match returns per-person max; swap-remove keeps mapping consistent', () => {
    const g = new GalleryIndex(4, 1);
    const e = (i: number) => l2normalize(Float32Array.from([0, 1, 2, 3].map((k) => (k === i ? 1 : 0.05))));
    g.upsertPhoto('A', 'a1', e(0));
    g.upsertPhoto('A', 'a2', e(1));
    g.upsertPhoto('B', 'b1', e(2));
    g.upsertPhoto('C', 'c1', e(3));
    expect(g.stats()).toEqual({ persons: 3, embeddings: 4 });
    expect(g.match(e(1), 2)[0]).toMatchObject({ personId: 'A', photoId: 'a2' });
    g.removePhoto('A', 'a1'); // last row (C) moves into slot 0
    expect(g.match(e(3), 1)[0]).toMatchObject({ personId: 'C', photoId: 'c1' });
    expect(g.match(e(1), 1)[0]).toMatchObject({ personId: 'A', photoId: 'a2' });
    g.setPersonActive('C', false);
    expect(g.match(e(3), 3).map((c) => c.personId)).not.toContain('C');
    expect(g.match(e(3), 1, { includeInactive: true })[0].personId).toBe('C');
    expect(g.match(e(1), 1, { exclude: 'A' })[0].personId).not.toBe('A');
    g.removePerson('A');
    expect(g.stats()).toEqual({ persons: 2, embeddings: 2 });
    g.upsertPhoto('D', 'd1', e(1));
    expect(g.match(e(1), 1)[0].personId).toBe('D');
    expect(g.scorePerson('D', e(1))).toBeCloseTo(1, 5);
  });

  it('randomized add/remove agrees with a naive reference', () => {
    const g = new GalleryIndex(16, 2);
    const ref = new Map<string, Float32Array>();
    for (let step = 0; step < 2000; step++) {
      const pid = `p${Math.floor(Math.random() * 20)}`;
      const key = `${pid}/ph${Math.floor(Math.random() * 4)}`;
      if (Math.random() < 0.6) {
        const v = rand(16);
        g.upsertPhoto(pid, key.split('/')[1], v);
        ref.set(key, v);
      } else {
        g.removePhoto(pid, key.split('/')[1]);
        ref.delete(key);
      }
    }
    const probe = rand(16);
    const best = new Map<string, number>();
    for (const [k, v] of ref) {
      const s = v.reduce((a, x, i) => a + x * probe[i], 0);
      const pid = k.split('/')[0];
      best.set(pid, Math.max(best.get(pid) ?? -Infinity, s));
    }
    const expected = [...best.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
    const got = g.match(probe, 5);
    expect(got.map((c) => c.personId)).toEqual(expected.map((e) => e[0]));
    got.forEach((c, i) => expect(c.score).toBeCloseTo(expected[i][1], 5));
    expect(g.stats().embeddings).toBe(ref.size);
  });

  it('matches 50k x 512 within the spec budget (≤ 50 ms, §8.6)', () => {
    const g = new GalleryIndex(512, 50_000);
    const v = rand(512);
    for (let i = 0; i < 50_000; i++) g.upsertPhoto(`p${i % 10000}`, `ph${i}`, v);
    const probe = rand(512);
    g.match(probe, 2); // warm-up (JIT)
    const t0 = performance.now();
    const n = 10;
    for (let i = 0; i < n; i++) g.match(probe, 2);
    const ms = (performance.now() - t0) / n;
    console.log(`[bench] match 50k x 512: ${ms.toFixed(1)} ms`);
    expect(ms).toBeLessThanOrEqual(50);
  });
});

describe('index cache', () => {
  it('rebuilds when the embedding list changes; loads cache otherwise', async () => {
    const d = tmp();
    const c = new AesGcmCipher(randomBytes(32));
    const store = new FileStore(d, c);
    await store.init();
    const log = pino({ level: 'silent' });
    const pid = '0192f0c4-0000-7000-8000-000000000001';
    const ph = '0192f0c4-0000-7000-8000-00000000000a';
    const v = rand(8);
    await store.writeEmbedding(pid, ph, 'm@1', v);
    const person = { id: pid, photos: [{ id: ph, modelKeys: ['m@1'] }], status: 'active' } as never;
    const m1 = new IndexManager(new GalleryIndex(8), 'm@1', store, log);
    expect((await m1.load([person])).source).toBe('rebuild');
    const m2 = new IndexManager(new GalleryIndex(8), 'm@1', store, log);
    expect((await m2.load([person])).source).toBe('cache');
    expect(m2.index.scorePerson(pid, v)).toBeCloseTo(1, 5);
    const m3 = new IndexManager(new GalleryIndex(8), 'm@1', store, log);
    expect((await m3.load([])).source).toBe('rebuild');
    expect(m3.index.stats().embeddings).toBe(0);
  });

  it('startup GC removes incomplete person directories and trash', async () => {
    const d = tmp();
    const store = new FileStore(d, new AesGcmCipher(randomBytes(32)));
    await store.init();
    const pid = '0192f0c4-0000-7000-8000-000000000002';
    await store.writeEmbedding(pid, '0192f0c4-0000-7000-8000-00000000000b', 'm@1', rand(8)); // no person.json
    writeFileSync(path.join(d, 'persons', '.trash-x'), '');
    const r = await store.init();
    expect(r.removedDirs.sort()).toEqual(['.trash-x', pid].sort());
    expect(existsSync(path.join(d, 'persons', pid))).toBe(false);
  });
});

describe('keyed mutex', () => {
  it('serializes per key and runs different keys concurrently', async () => {
    const m = new KeyedMutex();
    const log: string[] = [];
    const task = (k: string, id: string, ms: number) =>
      m.run(k, async () => {
        log.push(`start ${id}`);
        await new Promise((r) => setTimeout(r, ms));
        log.push(`end ${id}`);
      });
    await Promise.all([task('a', '1', 30), task('a', '2', 1), task('b', '3', 1)]);
    expect(log.indexOf('end 1')).toBeLessThan(log.indexOf('start 2'));
    expect(log.indexOf('start 3')).toBeLessThan(log.indexOf('end 1'));
  });
});

describe('config store', () => {
  it('persists patches but never runtime overrides', async () => {
    const d = tmp();
    const s = new ConfigStore(d, { privacy: { encryptAtRest: false } });
    expect(s.get().privacy.encryptAtRest).toBe(false);
    const r = await s.patch({ match: { acceptThreshold: 0.5 }, server: { port: 5000 } });
    expect(r.restartRequired).toEqual(['server.port']);
    const onDisk = JSON.parse(readFileSync(path.join(d, 'config.json'), 'utf8'));
    expect(onDisk).toEqual({ match: { acceptThreshold: 0.5 }, server: { port: 5000 } });
    await expect(s.patch({ match: { acceptThreshold: 'x' } })).rejects.toThrow();
    expect(deepMerge({ a: { b: 1, c: [1] } }, { a: { c: [2] } })).toEqual({ a: { b: 1, c: [2] } });
  });
});

describe('recognition journal', () => {
  it('pages newest-first by cursor, filters, purges and applies retention', async () => {
    const d = tmp();
    const j = new RecognitionJournal(d);
    const mk = (i: number, personId: string | undefined, ts: string) =>
      ({
        type: 'recognition.result', eventId: `0192f0c4-0000-7000-8000-${String(i).padStart(12, '0')}`, ts, sourceId: 's', trackId: 't',
        status: personId ? 'match' : 'unknown', person: personId ? { id: personId, firstName: 'x', lastName: 'y' } : undefined,
        score: 0.5, secondScore: 0.1, frameAgreement: 1, framesUsed: 5, attempt: 1, latencyMs: 100, liveness: null,
      }) as never;
    const today = new Date().toISOString();
    const old = new Date(Date.now() - 100 * 86400_000).toISOString();
    await j.append(mk(1, 'P1', old));
    for (let i = 2; i <= 6; i++) await j.append(mk(i, i % 2 ? 'P1' : undefined, today));
    const p1 = await j.query({ limit: 2 });
    expect(p1.items.map((e) => e.eventId.slice(-1))).toEqual(['6', '5']);
    const p2 = await j.query({ limit: 2, cursor: p1.nextCursor! });
    expect(p2.items.map((e) => e.eventId.slice(-1))).toEqual(['4', '3']);
    expect((await j.query({ limit: 10, personId: 'P1' })).items).toHaveLength(3);
    expect(JSON.stringify(await j.query({ limit: 10 }))).not.toContain('firstName');
    await j.applyRetention(90, 7);
    expect((await j.query({ limit: 10 })).items).toHaveLength(5);
    expect(await j.purgePerson('P1')).toBe(2);
    expect((await j.query({ limit: 10 })).items.every((e) => e.personId !== 'P1')).toBe(true);
  });
});
