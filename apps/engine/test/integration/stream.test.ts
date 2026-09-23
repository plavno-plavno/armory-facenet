// Stream pipeline end-to-end on a video file (spec §7, M3): events via WebSocket and webhooks, journal, snapshots.

import { createServer, type Server } from 'node:http';
import path from 'node:path';
import type { RecognitionEvent } from '@faceid/shared';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import WebSocket from 'ws';
import { signBody } from '../../src/core/webhooks.js';
import { consent, enrollPhoto, FIX, startEngine, waitFor, type TestEngine } from '../helpers.js';

let t: TestEngine;
let hookServer: Server;
const hooks: { body: string; sig: string }[] = [];
const wsEvents: any[] = [];
let ws: WebSocket;
let bushId: string;

beforeAll(async () => {
  hookServer = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      hooks.push({ body, sig: String(req.headers['x-signature']) });
      res.writeHead(hooks.length === 1 ? 500 : 200).end(); // first delivery fails -> retried
    });
  });
  await new Promise<void>((r) => hookServer.listen(0, '127.0.0.1', r));
  const hookPort = (hookServer.address() as { port: number }).port;
  t = await startEngine({
    config: {
      events: { storeSnapshots: true },
      webhooks: [{ url: `http://127.0.0.1:${hookPort}/hook`, secret: 'hook-secret-1', statuses: ['match'] }],
    },
  });
  const bush = await t.client.createPerson({ firstName: 'George', lastName: 'Bush', externalId: 'E-1', consent }, [
    enrollPhoto('George_W_Bush_0001.jpg'),
    enrollPhoto('George_W_Bush_0002.jpg'),
    enrollPhoto('George_W_Bush_0005.jpg'),
  ]);
  bushId = bush.id;
  await t.client.createPerson({ firstName: 'Colin', lastName: 'Powell', consent }, [enrollPhoto('Colin_Powell_0001.jpg')]);
  await t.client.createPerson({ firstName: 'Tony', lastName: 'Blair', consent }, [enrollPhoto('Tony_Blair_0001.jpg')]);

  ws = new WebSocket(t.baseUrl.replace('http', 'ws') + '/events', [`bearer.${t.token}`]);
  ws.on('message', (m) => wsEvents.push(JSON.parse(m.toString())));
  await waitFor(() => wsEvents.find((e) => e.type === 'auth.ok'), 5000);
});

afterAll(async () => {
  ws?.close();
  await t?.close();
  hookServer?.close();
});

describe('video file stream', () => {
  let results: RecognitionEvent[];

  it('recognizes the enrolled person and reports the stranger as unknown', async () => {
    const s = await t.client.createStream({
      name: 'debug-file',
      type: 'file',
      url: path.join(FIX, 'video/enrolled_then_stranger.mp4'),
      enabled: true,
      detectFps: 10,
    });
    // snapshot with overlay while running
    await waitFor(() => wsEvents.find((e) => e.type === 'stream.status' && e.status === 'running'), 15000);
    await new Promise((r) => setTimeout(r, 800));
    const snap = await fetch(`${t.baseUrl}/streams/${s.id}/snapshot?overlay=true`, { headers: { Authorization: `Bearer ${t.token}` } });
    expect(snap.status).toBe(200);
    expect(snap.headers.get('content-type')).toBe('image/jpeg');

    await waitFor(() => wsEvents.find((e) => e.type === 'stream.status' && e.status === 'stopped'), 30000);
    await new Promise((r) => setTimeout(r, 1500)); // let in-flight identifications finish
    results = wsEvents.filter((e) => e.type === 'recognition.result');
    const summary = results.map((e) => `${e.trackId}:${e.status}:${e.person?.lastName ?? ''}:${e.score}:${e.latencyMs}ms`);
    console.log('[stream] events:', summary.join(' | '));

    const matches = results.filter((e) => e.status === 'match');
    expect(matches).toHaveLength(1);
    expect(matches[0].person).toMatchObject({ id: bushId, externalId: 'E-1', firstName: 'George', lastName: 'Bush' });
    expect(matches[0].latencyMs).toBeLessThanOrEqual(1500);
    expect(matches[0].framesUsed).toBeGreaterThanOrEqual(3);
    expect(matches[0].frameAgreement).toBeGreaterThanOrEqual(0.6);
    expect(matches[0].liveness).toBeNull();

    // The stranger's track must end as unknown, never as a match.
    const strangerTrack = results.find((e) => e.trackId !== matches[0].trackId && e.status !== 'low_quality')?.trackId;
    expect(strangerTrack).toBeTruthy();
    const finalStranger = results.filter((e) => e.trackId === strangerTrack).at(-1)!;
    expect(finalStranger.status).toBe('unknown');
    expect(finalStranger.person).toBeUndefined();
  });

  it('writes the recognition journal (identifiers only)', async () => {
    const j = await t.client.recognitions({ status: 'match' });
    expect(j.items).toHaveLength(1);
    expect(j.items[0]).toMatchObject({ personId: bushId, status: 'match' });
    expect(JSON.stringify(j.items[0])).not.toContain('Bush');
    const all = await t.client.recognitions({ limit: '1' });
    expect(all.items).toHaveLength(1);
    expect(all.nextCursor).toBeTruthy();
  });

  it('stores encrypted snapshots retrievable by id', async () => {
    const m = results.find((e) => e.status === 'match')!;
    expect(m.snapshotId).toBeTruthy();
    const r = await fetch(`${t.baseUrl}/events/snapshots/${m.snapshotId}`, { headers: { Authorization: `Bearer ${t.token}` } });
    expect(r.status).toBe(200);
    expect(Buffer.from(await r.arrayBuffer()).subarray(0, 2)).toEqual(Buffer.from([0xff, 0xd8]));
  });

  it('delivers signed webhooks with retry, filtered by status', async () => {
    await waitFor(() => hooks.length >= 2, 10000);
    for (const h of hooks) {
      expect(h.sig).toBe(signBody('hook-secret-1', h.body));
      expect(JSON.parse(h.body).status).toBe('match');
    }
    expect(JSON.parse(hooks[0].body).eventId).toBe(JSON.parse(hooks[1].body).eventId); // retry of the same event
  });

  it('health exposes latency metrics', async () => {
    const h = await t.client.health();
    expect(h.metrics.latencyP50).toBeGreaterThan(0);
  });

  it('purgeEvents removes journal lines and snapshots of the person', async () => {
    await t.client.deletePerson(bushId, true);
    expect((await t.client.recognitions({ personId: bushId })).items).toEqual([]);
    const m = results.find((e) => e.status === 'match')!;
    const r = await fetch(`${t.baseUrl}/events/snapshots/${m.snapshotId}`, { headers: { Authorization: `Bearer ${t.token}` } });
    expect(r.status).toBe(404);
  });

  it('WebSocket without a token is closed', async () => {
    const bad = new WebSocket(t.baseUrl.replace('http', 'ws') + '/events');
    const code = await new Promise<number>((resolve) => {
      bad.on('open', () => bad.send(JSON.stringify({ type: 'auth', token: 'wrong' })));
      bad.on('close', (c) => resolve(c));
    });
    expect(code).toBe(4401);
  });

  it('captures frames from a running stream and enrolls a person with them (source=camera)', async () => {
    const s = await t.client.createStream({ name: 'loop', type: 'file', url: path.join(FIX, 'video/enrolled_then_stranger.mp4'), enabled: true, loop: true });
    await waitFor(() => wsEvents.find((e) => e.type === 'stream.status' && e.sourceId === s.id && e.status === 'running'), 15000);
    const shot = await t.client.captureFromStream(s.id);
    expect(shot.frames.length).toBe(3); // enroll.captureFrames
    expect(shot.frames[0].width).toBe(1280);
    const blobs = shot.frames.map((f) => new Blob([Buffer.from(f.jpeg, 'base64')], { type: 'image/jpeg' }));
    const fd = new FormData();
    fd.append('data', JSON.stringify({ firstName: 'Cam', lastName: 'Captured', consent }));
    fd.append('photoSources', JSON.stringify(['camera', 'camera', 'upload']));
    blobs.forEach((b, i) => fd.append('photos', b, `c${i}.jpg`));
    const p = await t.client.request<{ photos: { source: string }[] }>('POST', '/persons?allowDuplicate=true', fd);
    expect(p.photos.map((x) => x.source)).toEqual(['camera', 'camera', 'upload']);
    const bad = new FormData();
    bad.append('data', JSON.stringify({ firstName: 'X', lastName: 'Y', consent }));
    bad.append('photoSources', JSON.stringify(['camera']));
    blobs.slice(0, 2).forEach((b, i) => bad.append('photos', b, `c${i}.jpg`));
    const r = await fetch(`${t.baseUrl}/persons`, { method: 'POST', body: bad, headers: { Authorization: `Bearer ${t.token}` } });
    expect(r.status).toBe(400);
    await t.client.deleteStream(s.id);
  });

  it('recognizes on demand ("Recognize"): match for an enrolled face, journaled as manual', async () => {
    const s = await t.client.createStream({ name: 'loop2', type: 'file', url: path.join(FIX, 'video/enrolled_then_stranger.mp4'), enabled: true, loop: true });
    await waitFor(() => wsEvents.find((e) => e.type === 'stream.status' && e.sourceId === s.id && e.status === 'running'), 15000);
    // The looped video alternates the enrolled face, gaps and a stranger: retry until the enrolled face is in view.
    let match: Awaited<ReturnType<typeof t.client.recognizeOnStream>> | null = null;
    const seen: string[] = [];
    for (let i = 0; i < 8 && !match; i++) {
      try {
        const r = await t.client.recognizeOnStream(s.id);
        seen.push(r.event.status);
        expect(r.face.length).toBeGreaterThan(100);
        if (r.event.status === 'match') match = r;
      } catch (e) {
        seen.push((e as { code: string }).code);
        expect(['NO_FACE', 'LOW_QUALITY']).toContain((e as { code: string }).code);
      }
    }
    console.log('[recognize] attempts:', seen.join(', '));
    expect(match).not.toBeNull();
    expect(match!.event.person?.lastName).toBe('Captured');
    expect(match!.event.trackId).toMatch(/^manual-/);
    expect(match!.candidates[0].personId).toBe(match!.event.person!.id);
    expect(match!.event.latencyMs).toBeLessThan(6000);
    const j = await t.client.recognitions({ personId: match!.event.person!.id });
    expect(j.items.some((x) => x.eventId === match!.event.eventId)).toBe(true);
    await t.client.deleteStream(s.id);
  });
});
