// WebSocket push + recognition journal + snapshots (spec §10.5).

import { promises as fs } from 'node:fs';
import type { EngineEvent } from '@faceid/shared';
import { RecognitionStatus } from '@faceid/shared';
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { Engine } from '../../engine.js';
import { ApiError } from '../../util/errors.js';

export const eventRoutes = (engine: Engine) => async (app: FastifyInstance) => {
  app.get('/events', { websocket: true, schema: { tags: ['events'], hide: true } } as object, (socket: any, req: any) => {
    let authed = false;
    let filter: Set<string> | null = null;
    let unsubscribe: (() => void) | null = null;
    const send = (m: unknown) => socket.readyState === 1 && socket.send(JSON.stringify(m));
    const onAuth = (actor: string) => {
      authed = true;
      send({ type: 'auth.ok', actor });
      unsubscribe = engine.bus.subscribe((e: EngineEvent) => {
        if (!filter || filter.has(e.type)) send(e);
      });
    };
    const proto = String(req.headers['sec-websocket-protocol'] ?? '')
      .split(',')
      .map((s) => s.trim())
      .find((s) => s.startsWith('bearer.'));
    const headerActor = engine.tokens.fromHeader(req.headers.authorization) ?? (proto ? engine.tokens.verify(proto.slice(7)) : null);
    if (headerActor) onAuth(headerActor);
    const authTimer = setTimeout(() => {
      if (!authed) socket.close(4401, 'unauthorized');
    }, 5000);
    socket.on('message', (raw: Buffer) => {
      let msg: any;
      try {
        msg = JSON.parse(raw.toString());
      } catch {
        return send({ type: 'error', error: { code: 'VALIDATION_ERROR', message: 'invalid JSON' } });
      }
      if (!authed) {
        const actor = msg?.type === 'auth' ? engine.tokens.verify(msg.token) : null;
        if (!actor) return socket.close(4401, 'unauthorized');
        return onAuth(actor);
      }
      if (msg?.type === 'subscribe' && Array.isArray(msg.events)) filter = new Set(msg.events.map(String));
    });
    socket.on('close', () => {
      clearTimeout(authTimer);
      unsubscribe?.();
    });
  });

  app.get(
    '/events/recognitions',
    {
      schema: {
        tags: ['events'],
        querystring: z.object({
          from: z.string().optional(),
          to: z.string().optional(),
          personId: z.string().optional(),
          sourceId: z.string().optional(),
          status: RecognitionStatus.optional(),
          limit: z.coerce.number().int().min(1).max(1000).default(100),
          cursor: z.string().optional(),
        }),
      },
    },
    async (req) => engine.journal.query(req.query as any),
  );

  app.get('/events/snapshots/:snapshotId', { schema: { tags: ['events'], params: z.object({ snapshotId: z.string().uuid() }) } }, async (req, reply) => {
    const file = await engine.journal.findSnapshot((req.params as { snapshotId: string }).snapshotId);
    if (!file) throw new ApiError('NOT_FOUND', 'Snapshot not found');
    return reply.header('Content-Type', 'image/jpeg').header('Cache-Control', 'no-store').send(engine.cipher.decrypt(await fs.readFile(file)));
  });
};
