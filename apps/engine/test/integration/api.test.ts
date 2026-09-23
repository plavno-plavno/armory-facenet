// API integration tests: every endpoint of spec §10 with its error codes (§10.1, §15.2, §15.4).

import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { ApiClientError, type Person } from '@faceid/shared';
import { AesGcmCipher } from '../../src/store/crypto.js';
import { consent, enrollPhoto, FIX, fixtureBlob, startEngine, type TestEngine } from '../helpers.js';

let t: TestEngine;
const logs: string[] = [];
let bush: Person;
let powell: Person;

async function expectApiError(p: Promise<unknown>, status: number, code: string): Promise<ApiClientError> {
  try {
    await p;
  } catch (e) {
    expect(e).toBeInstanceOf(ApiClientError);
    const err = e as ApiClientError;
    expect({ status: err.status, code: err.code, message: err.message }).toMatchObject({ status, code });
    return err;
  }
  throw new Error(`expected ${status} ${code}, request succeeded`);
}

async function raw(method: string, url: string, init: RequestInit = {}) {
  return fetch(`${t.baseUrl}${url}`, { method, ...init, headers: { Authorization: `Bearer ${t.token}`, ...(init.headers ?? {}) } });
}

async function jpegOf(width: number, height: number, blurSigma = 0): Promise<Blob> {
  let img = sharp(readFileSync(path.join(FIX, 'enroll', 'George_W_Bush_0002.jpg'))).resize(width, height);
  if (blurSigma) img = img.blur(blurSigma);
  return new Blob([await img.jpeg().toBuffer()], { type: 'image/jpeg' });
}

function listFiles(dir: string): string[] {
  const out: string[] = [];
  for (const f of readdirSync(dir)) {
    const p = path.join(dir, f);
    if (statSync(p).isDirectory()) out.push(...listFiles(p));
    else out.push(p);
  }
  return out;
}

beforeAll(async () => {
  t = await startEngine({ logs });
});
afterAll(async () => {
  await t?.close();
});

describe('auth & docs', () => {
  it('rejects requests without a valid token', async () => {
    const r = await fetch(`${t.baseUrl}/health`);
    expect(r.status).toBe(401);
    expect((await r.json()).error.code).toBe('UNAUTHORIZED');
    const r2 = await fetch(`${t.baseUrl}/health`, { headers: { Authorization: 'Bearer nope' } });
    expect(r2.status).toBe(401);
  });

  it('serves OpenAPI generated from zod schemas', async () => {
    const r = await fetch(`${t.baseUrl}/openapi.json`);
    expect(r.status).toBe(200);
    const doc = await r.json();
    expect(doc.openapi).toMatch(/^3\./);
    for (const p of ['/persons', '/persons/{id}', '/persons/{id}/photos', '/recognize/identify', '/streams', '/health', '/config', '/admin/reindex']) {
      expect(Object.keys(doc.paths)).toContain(p);
    }
  });

  it('unknown routes -> 404 NOT_FOUND', async () => {
    const r = await raw('GET', '/nope');
    expect(r.status).toBe(404);
    expect((await r.json()).error.code).toBe('NOT_FOUND');
  });
});

describe('enrollment rules (§9)', () => {
  it('creates a person with 3 photos', async () => {
    bush = await t.client.createPerson(
      { firstName: 'George', lastName: 'Bush', externalId: 'E-1042', consent, customFields: { badge: 'A-17' } },
      ['George_W_Bush_0001.jpg', 'George_W_Bush_0002.jpg', 'George_W_Bush_0005.jpg'].map(enrollPhoto),
    );
    expect(bush.photos).toHaveLength(3);
    expect(bush.version).toBe(1);
    expect(bush.status).toBe('active');
    expect(bush.photos[0].modelKeys).toEqual([t.engine.indexMgr.modelKey]);
    expect(bush.consent?.obtainedAt).toBeTruthy();
    powell = await t.client.createPerson({ firstName: 'Colin', lastName: 'Powell', consent }, [enrollPhoto('Colin_Powell_0001.jpg'), enrollPhoto('Colin_Powell_0002.jpg')]);
    expect(t.engine.index.stats()).toEqual({ persons: 2, embeddings: 5 });
  });

  it('CONSENT_REQUIRED', async () => {
    await expectApiError(t.client.createPerson({ firstName: 'A', lastName: 'B' }, [enrollPhoto('Tony_Blair_0001.jpg')]), 400, 'CONSENT_REQUIRED');
  });

  it('VALIDATION_ERROR for bad person data', async () => {
    await expectApiError(t.client.createPerson({ firstName: 'A' } as never, [enrollPhoto('Tony_Blair_0001.jpg')]), 400, 'VALIDATION_ERROR');
  });

  it('externalId must be unique', async () => {
    await expectApiError(
      t.client.createPerson({ firstName: 'A', lastName: 'B', externalId: 'E-1042', consent }, [enrollPhoto('Tony_Blair_0001.jpg')]),
      409,
      'VALIDATION_ERROR',
    );
  });

  it('NO_FACE', async () => {
    await expectApiError(t.client.createPerson({ firstName: 'A', lastName: 'B', consent }, [fixtureBlob('parity/noface_gradient.png', 'image/png')]), 422, 'NO_FACE');
  });

  it('MULTIPLE_FACES returns boxes; faceBox selects a face', async () => {
    const err = await expectApiError(t.client.createPerson({ firstName: 'A', lastName: 'B', consent }, [fixtureBlob('raw/t1.jpg')]), 422, 'MULTIPLE_FACES');
    const boxes = err.details.boxes as number[][];
    expect(boxes.length).toBeGreaterThan(1);
    const fd = new FormData();
    fd.append('data', JSON.stringify({ firstName: 'A', lastName: 'B', consent }));
    fd.append('faceBox', JSON.stringify(boxes[0]));
    fd.append('photos', fixtureBlob('raw/t1.jpg'), 't1.jpg');
    const r = await raw('POST', '/persons', { body: fd });
    expect((await r.json()).error?.code).not.toBe('MULTIPLE_FACES');
  });

  it('UNSUPPORTED_FORMAT', async () => {
    await expectApiError(
      t.client.createPerson({ firstName: 'A', lastName: 'B', consent }, [new Blob(['not an image'], { type: 'text/plain' })]),
      415,
      'UNSUPPORTED_FORMAT',
    );
  });

  it('IMAGE_TOO_SMALL', async () => {
    await expectApiError(t.client.createPerson({ firstName: 'A', lastName: 'B', consent }, [await jpegOf(150, 150)]), 422, 'IMAGE_TOO_SMALL');
  });

  it('LOW_QUALITY lists failed metrics', async () => {
    const err = await expectApiError(t.client.createPerson({ firstName: 'A', lastName: 'B', consent }, [await jpegOf(500, 500, 6)]), 422, 'LOW_QUALITY');
    expect(err.details.reasons).toContain('sharpness');
  });

  it('FILE_TOO_LARGE', async () => {
    const big = new Blob([Buffer.alloc(11 * 1024 * 1024, 1)], { type: 'image/jpeg' });
    await expectApiError(t.client.createPerson({ firstName: 'A', lastName: 'B', consent }, [big]), 413, 'FILE_TOO_LARGE');
  });

  it('DUPLICATE_SUSPECTED unless allowDuplicate', async () => {
    const err = await expectApiError(
      t.client.createPerson({ firstName: 'X', lastName: 'Y', consent }, [enrollPhoto('George_W_Bush_0005.jpg')]),
      409,
      'DUPLICATE_SUSPECTED',
    );
    expect((err.details.candidate as { personId: string }).personId).toBe(bush.id);
    const dup = await t.client.createPerson({ firstName: 'X', lastName: 'Y', consent }, [enrollPhoto('George_W_Bush_0005.jpg')], { allowDuplicate: true });
    await t.client.deletePerson(dup.id);
    const audit = readFileSync(path.join(t.dataDir, 'logs/audit', `${new Date().toISOString().slice(0, 10)}.jsonl`), 'utf8');
    expect(audit).toMatch(/"flags":\["allowDuplicate"\]/);
  });

  it('PHOTO_MISMATCH on photos of another person (create and add), force bypasses', async () => {
    await expectApiError(
      t.client.createPerson({ firstName: 'Z', lastName: 'Z', consent }, [enrollPhoto('Tony_Blair_0001.jpg'), enrollPhoto('Colin_Powell_0003.jpg')]),
      422,
      'PHOTO_MISMATCH',
    );
    const err = await expectApiError(t.client.addPhotos(bush.id, [enrollPhoto('Tony_Blair_0001.jpg')]), 422, 'PHOTO_MISMATCH');
    expect(typeof err.details.score).toBe('number');
    const added = (await t.client.addPhotos(bush.id, [enrollPhoto('Tony_Blair_0001.jpg')], { force: true })) as { id: string }[];
    expect(added).toHaveLength(1);
    await t.client.deletePhoto(bush.id, added[0].id);
  });

  it('creation is atomic: one bad photo -> nothing stored', async () => {
    const before = readdirSync(path.join(t.dataDir, 'persons')).length;
    const err = await expectApiError(
      t.client.createPerson({ firstName: 'Tony', lastName: 'Blair', consent }, [enrollPhoto('Tony_Blair_0001.jpg'), fixtureBlob('parity/noface_noise.png', 'image/png')]),
      422,
      'NO_FACE',
    );
    expect((err.details.photos as { index: number }[])[0].index).toBe(1);
    expect(readdirSync(path.join(t.dataDir, 'persons')).length).toBe(before);
    expect(t.engine.persons.count()).toBe(2);
  });
});

describe('persons CRUD (§10.3)', () => {
  it('lists with search, status filter and cursor paging', async () => {
    expect((await t.client.listPersons({ q: 'bush' })).items.map((p) => p.id)).toEqual([bush.id]);
    expect((await t.client.listPersons({ q: 'E-1042' })).total).toBe(1);
    const p1 = await t.client.listPersons({ limit: 1 });
    expect(p1.total).toBe(2);
    expect(p1.nextCursor).toBeTruthy();
    const p2 = await t.client.listPersons({ limit: 1, cursor: p1.nextCursor! });
    expect(p2.items[0].id).not.toBe(p1.items[0].id);
    expect(p2.nextCursor).toBeNull();
  });

  it('GET by id and 404', async () => {
    expect((await t.client.getPerson(bush.id)).lastName).toBe('Bush');
    await expectApiError(t.client.getPerson('0192f0c4-0000-7000-8000-000000000000'), 404, 'NOT_FOUND');
  });

  it('PATCH with optimistic locking', async () => {
    const cur = await t.client.getPerson(bush.id);
    const upd = await t.client.patchPerson(bush.id, { position: 'President', notes: 'n' }, cur.version);
    expect(upd.version).toBe(cur.version + 1);
    expect(upd.position).toBe('President');
    await expectApiError(t.client.patchPerson(bush.id, { position: 'X' }, cur.version), 412, 'VERSION_CONFLICT');
    const cleared = await t.client.patchPerson(bush.id, { notes: null });
    expect(cleared.notes).toBeUndefined();
    await expectApiError(t.client.patchPerson(bush.id, { unknownField: 1 } as never), 400, 'VALIDATION_ERROR');
  });

  it('disabled persons are not matched', async () => {
    await t.client.patchPerson(bush.id, { status: 'disabled' });
    const probe = enrollPhoto('George_W_Bush_0002.jpg');
    const idf = new FormData();
    idf.append('photo', probe, 'p.jpg');
    let r = await (await raw('POST', '/recognize/identify', { body: idf })).json();
    expect(r.faces[0].candidates.map((c: { personId: string }) => c.personId)).not.toContain(bush.id);
    await t.client.patchPerson(bush.id, { status: 'active' });
    const idf2 = new FormData();
    idf2.append('photo', probe, 'p.jpg');
    r = await (await raw('POST', '/recognize/identify', { body: idf2 })).json();
    expect(r.faces[0].candidates[0].personId).toBe(bush.id);
  });

  it('photos: list, download original/aligned, LAST_PHOTO', async () => {
    const photos = (await (await raw('GET', `/persons/${powell.id}/photos`)).json()) as { id: string }[];
    expect(photos).toHaveLength(2);
    const orig = await raw('GET', `/persons/${powell.id}/photos/${photos[0].id}?variant=original`);
    expect(orig.headers.get('content-type')).toBe('image/jpeg');
    const meta = await sharp(Buffer.from(await orig.arrayBuffer())).metadata();
    expect(meta.width).toBe(500);
    expect(meta.exif).toBeUndefined();
    const al = await raw('GET', `/persons/${powell.id}/photos/${photos[0].id}?variant=aligned`);
    expect((await sharp(Buffer.from(await al.arrayBuffer())).metadata()).width).toBe(112);
    await t.client.deletePhoto(powell.id, photos[1].id);
    await expectApiError(t.client.deletePhoto(powell.id, photos[0].id), 409, 'LAST_PHOTO');
    await expectApiError(t.client.deletePhoto(powell.id, '0192f0c4-0000-7000-8000-000000000000'), 404, 'NOT_FOUND');
  });

  it('export returns a zip with person.json and originals', async () => {
    const r = await raw('GET', `/persons/${bush.id}/export`);
    expect(r.headers.get('content-type')).toBe('application/zip');
    const buf = Buffer.from(await r.arrayBuffer());
    expect(buf.subarray(0, 2).toString()).toBe('PK');
    expect(buf.includes(Buffer.from('person.json'))).toBe(true);
    expect(buf.includes(Buffer.from('photos/'))).toBe(true);
  });
});

describe('one-shot recognition (§10.4)', () => {
  const identify = async (blob: Blob) => {
    const fd = new FormData();
    fd.append('photo', blob, 'p.jpg');
    fd.append('topK', '3');
    return (await raw('POST', '/recognize/identify', { body: fd })).json();
  };

  it('identify: enrolled person -> match, stranger -> unknown', async () => {
    const r1 = await identify(await jpegOfPerson('George_W_Bush', 20));
    // LFW frames may contain background faces; the subject is the largest one.
    const main = r1.faces.reduce((a: any, b: any) => (b.box[2] * b.box[3] > a.box[2] * a.box[3] ? b : a));
    expect(main.candidates[0].personId).toBe(bush.id);
    expect(main.status).toBe('match');
    const r2 = await identify(await jpegOfPerson('Gerhard_Schroeder', 1));
    expect(r2.faces[0].status).toBe('unknown');
    const r3 = await identify(fixtureBlob('parity/noface_noise.png', 'image/png'));
    expect(r3.faces).toEqual([]);
  });

  it('verify', async () => {
    const fd = new FormData();
    fd.append('photo', await jpegOfPerson('George_W_Bush', 20), 'p.jpg');
    fd.append('personId', bush.id);
    const r = await (await raw('POST', '/recognize/verify', { body: fd })).json();
    expect(r.match).toBe(true);
    expect(r.threshold).toBe(0.35);
    const fd2 = new FormData();
    fd2.append('photo', await jpegOfPerson('George_W_Bush', 20), 'p.jpg');
    fd2.append('personId', powell.id);
    expect((await (await raw('POST', '/recognize/verify', { body: fd2 })).json()).match).toBe(false);
  });
});

async function jpegOfPerson(person: string, idx: number): Promise<Blob> {
  const f = path.join(FIX, 'raw/lfw', person, `${person}_${String(idx).padStart(4, '0')}.jpg`);
  return new Blob([await sharp(readFileSync(f)).resize(500, 500).jpeg({ quality: 95 }).toBuffer()], { type: 'image/jpeg' });
}

describe('streams API (§10.5)', () => {
  it('validation, 404 and SOURCE_UNAVAILABLE', async () => {
    await expectApiError(t.client.createStream({ name: 'x', type: 'rtsp', enabled: false }), 400, 'VALIDATION_ERROR');
    await expectApiError(t.client.streamAction('0192f0c4-0000-7000-8000-000000000000', 'start'), 404, 'NOT_FOUND');
    const s = await t.client.createStream({ name: 'cam', type: 'rtsp', url: 'rtsp://user:secret@10.255.255.1/stream', enabled: false });
    expect(s.url).toBe('rtsp://***@10.255.255.1/stream');
    expect(s.status).toBe('stopped');
    const r = await raw('GET', `/streams/${s.id}/snapshot`);
    expect(r.status).toBe(409);
    expect((await r.json()).error.code).toBe('SOURCE_UNAVAILABLE');
    await expectApiError(t.client.capturePhotos(bush.id, s.id), 409, 'SOURCE_UNAVAILABLE');
    const p = await t.client.request<{ name: string }>('PATCH', `/streams/${s.id}`, { name: 'cam2' });
    expect(p.name).toBe('cam2');
    expect((await t.client.listStreams()).map((x) => x.id)).toContain(s.id);
    await t.client.deleteStream(s.id);
    expect(await t.client.listStreams()).toEqual([]);
    // the RTSP URL (with credentials) is stored encrypted
    expect(readFileSync(path.join(t.dataDir, 'streams.json.enc')).includes(Buffer.from('secret'))).toBe(false);
  });

  it('cameras list and journal endpoints respond', async () => {
    expect(Array.isArray(await t.client.cameras())).toBe(true);
    expect(await t.client.recognitions()).toEqual({ items: [], nextCursor: null });
    const r = await raw('GET', '/events/snapshots/0192f0c4-0000-7000-8000-000000000000');
    expect(r.status).toBe(404);
  });
});

describe('system & admin (§10.6)', () => {
  it('health', async () => {
    const h = await t.client.health();
    expect(h.status).toBe('ok');
    expect(h.models.embedder.modelKey).toBe('lvface-s-glint360k@1');
    expect(h.gallery).toEqual({ persons: 2, embeddings: 4 });
    expect(h.metrics).toHaveProperty('latencyP95');
  });

  it('config: read, patch, validation, restart-required keys, masked webhook secrets', async () => {
    const c = await t.client.getConfig();
    expect(c.config.match.acceptThreshold).toBe(0.35);
    const p = await t.client.patchConfig({ pipeline: { cooldownMs: 5000 }, webhooks: [{ url: 'http://127.0.0.1:9/hook', secret: 'supersecret' }] });
    expect(p.config.pipeline.cooldownMs).toBe(5000);
    expect(p.config.webhooks[0].secret).toBe('***');
    expect(t.engine.cfg().webhooks[0].secret).toBe('supersecret');
    await expectApiError(t.client.patchConfig({ match: { acceptThreshold: 7 } }), 400, 'VALIDATION_ERROR');
    const r = await t.client.patchConfig({ models: { embedder: 'lvface-t-glint360k' } });
    expect(r.restartRequired).toContain('models.embedder');
    await t.client.patchConfig({ models: { embedder: 'lvface-s-glint360k' }, webhooks: [] });
  });

  it('reindex job runs to completion', async () => {
    const { jobId } = await t.client.request<{ jobId: string }>('POST', '/admin/reindex');
    for (let i = 0; i < 100; i++) {
      const j = await t.client.job(jobId);
      if (j.state !== 'running') {
        expect(j).toMatchObject({ state: 'completed', done: 4, total: 4, errors: [] });
        break;
      }
      await new Promise((r) => setTimeout(r, 200));
    }
    await expectApiError(t.client.job('nope'), 404, 'NOT_FOUND');
  });

  it('calibration gallery scores', async () => {
    const r = await t.client.request<any>('GET', '/admin/calibration/gallery-scores');
    expect(r.genuine.n).toBe(3); // 3 photos of Bush -> 3 pairs; Powell has 1
    expect(r.impostor.n).toBe(3);
    expect(r.genuine.mean).toBeGreaterThan(r.impostor.mean);
  });

  it('token rotation invalidates the old token', async () => {
    const { token } = await t.client.request<{ token: string }>('POST', '/admin/token/rotate');
    expect((await fetch(`${t.baseUrl}/health`, { headers: { Authorization: `Bearer ${t.token}` } })).status).toBe(401);
    expect((await fetch(`${t.baseUrl}/health`, { headers: { Authorization: `Bearer ${token}` } })).status).toBe(200);
    t.token = token;
    (t.client as unknown as { opts: { token: string } }).opts.token = token;
  });
});

describe('privacy & security (§12, §15.4 item 5, item 7)', () => {
  it('all *.enc files are unreadable without the key', () => {
    const files = listFiles(t.dataDir).filter((f) => f.endsWith('.enc'));
    expect(files.length).toBeGreaterThan(5);
    const wrong = new AesGcmCipher(Buffer.alloc(32, 7));
    for (const f of files) {
      const buf = readFileSync(f);
      expect(buf.subarray(0, 4).toString()).toBe('FENC');
      expect(buf.includes(Buffer.from('Bush'))).toBe(false);
      expect(() => wrong.decrypt(buf)).toThrow();
    }
  });

  it('application logs contain no personal data', () => {
    const appLogs = logs.join('\n');
    for (const pii of ['George', 'Bush', 'Colin', 'Powell', 'E-1042', 'A-17', 'President']) expect(appLogs).not.toContain(pii);
  });

  it('DELETE removes the person from recognition and disk immediately', async () => {
    const dir = path.join(t.dataDir, 'persons', bush.id);
    expect(existsSync(dir)).toBe(true);
    await t.client.deletePerson(bush.id, true);
    expect(existsSync(dir)).toBe(false);
    expect(readdirSync(path.join(t.dataDir, 'persons')).some((n) => n.includes(bush.id))).toBe(false);
    const fd = new FormData();
    fd.append('photo', await jpegOfPerson('George_W_Bush', 20), 'p.jpg');
    const r = await (await raw('POST', '/recognize/identify', { body: fd })).json();
    expect(r.faces.flatMap((f: any) => f.candidates.map((c: { personId: string }) => c.personId))).not.toContain(bush.id);
    await expectApiError(t.client.getPerson(bush.id), 404, 'NOT_FOUND');
    await expectApiError(t.client.deletePerson(bush.id), 404, 'NOT_FOUND');
  });
});

describe('restart & index cache (§8.3, §15.4 item 6)', () => {
  it('gallery survives restart via cache; corrupted cache is rebuilt; wrong key fails', async () => {
    const extra = await t.client.createPerson({ firstName: 'Tony', lastName: 'Blair', consent }, [enrollPhoto('Tony_Blair_0001.jpg'), enrollPhoto('Tony_Blair_0002.jpg')]);
    const statsBefore = t.engine.index.stats();
    const { dataDir, dek } = t;
    await t.close({ keepData: true });

    t = await startEngine({ dataDir, dek, logs });
    expect(t.engine.index.stats()).toEqual(statsBefore);
    expect((await t.client.getPerson(extra.id)).lastName).toBe('Blair');

    await t.close({ keepData: true });
    const bin = readdirSync(path.join(dataDir, 'index')).find((f) => f.endsWith('.bin.enc'))!;
    writeFileSync(path.join(dataDir, 'index', bin), Buffer.from('garbage'));
    t = await startEngine({ dataDir, dek, logs });
    expect(t.engine.index.stats()).toEqual(statsBefore);
    expect(logs.join('\n')).toMatch(/gallery cache is unreadable, rebuilding/);
    await t.close({ keepData: true });

    await expect(startEngine({ dataDir, dek: Buffer.alloc(32, 1), logs })).rejects.toThrow();
    t = await startEngine({ dataDir, dek, logs });
  });
});
