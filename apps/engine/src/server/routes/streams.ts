// Streams & cameras (spec §10.5).

import { StreamInfo, StreamInput } from '@faceid/shared';
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import { publicQuality } from '../../core/analyzer.js';
import type { Engine } from '../../engine.js';
import { BoolFlag, errors, IdParams } from './common.js';

export const streamRoutes = (engine: Engine) => async (app: FastifyInstance) => {
  const sm = engine.streams;
  const id = (req: { params: unknown }) => (req.params as { id: string }).id;

  app.get('/devices/cameras', { schema: { tags: ['streams'], response: { 200: z.array(z.object({ deviceId: z.string(), label: z.string() })) } } }, async () =>
    sm.listCameras(),
  );
  app.get('/streams', { schema: { tags: ['streams'], response: { 200: z.array(StreamInfo) } } }, async () => sm.list());
  app.post('/streams', { schema: { tags: ['streams'], body: StreamInput, response: { 201: StreamInfo, ...errors(400) } } }, async (req, reply) =>
    reply.status(201).send(await sm.create(req.body as StreamInput)),
  );
  app.get('/streams/:id', { schema: { tags: ['streams'], params: IdParams, response: { 200: StreamInfo, ...errors(404) } } }, async (req) => sm.info(id(req)));
  app.patch(
    '/streams/:id',
    { schema: { tags: ['streams'], params: IdParams, body: StreamInput.partial(), response: { 200: StreamInfo, ...errors(400, 404) } } },
    async (req) => sm.patch(id(req), req.body as Partial<StreamInput>),
  );
  app.delete('/streams/:id', { schema: { tags: ['streams'], params: IdParams } }, async (req, reply) => {
    await sm.remove(id(req));
    return reply.status(204).send();
  });
  app.post('/streams/:id/start', { schema: { tags: ['streams'], params: IdParams, response: { 200: StreamInfo, ...errors(404) } } }, async (req) => sm.start(id(req)));
  app.post('/streams/:id/stop', { schema: { tags: ['streams'], params: IdParams, response: { 200: StreamInfo, ...errors(404) } } }, async (req) => sm.stop(id(req)));
  // Capture enrollment-quality frames from a running stream without creating anything (admin UI
  // "take photo from camera" during registration). Same rules as spec §9.3; nothing is stored.
  app.post(
    '/streams/:id/capture',
    {
      config: { rateLimit: { max: engine.cfg().server.enrollRatePerSec, timeWindow: 1000 } },
      schema: {
        tags: ['streams'],
        summary: 'Capture the best enroll.captureFrames frames with exactly one face (JPEG, base64)',
        params: IdParams,
        response: {
          200: z.object({
            frames: z.array(z.object({ jpeg: z.string(), width: z.number(), height: z.number(), faceBox: z.array(z.number()), quality: z.record(z.string(), z.number()) })),
          }),
          ...errors(404, 409, 422),
        },
      },
    },
    async (req) => {
      const shots = await sm.capture(id(req));
      return {
        frames: shots.map((a) => ({
          jpeg: a.originalJpeg.toString('base64'),
          width: a.frame.width,
          height: a.frame.height,
          faceBox: a.det.box.map((v) => Math.round(v)),
          quality: publicQuality(a.quality),
        })),
      };
    },
  );

  app.post(
    '/streams/:id/recognize',
    {
      config: { rateLimit: { max: 5, timeWindow: 1000 } },
      schema: {
        tags: ['streams'],
        summary: 'On-demand recognition of the person in front of the camera (burst + decision, §7.5–7.7)',
        params: IdParams,
        response: errors(404, 409, 422),
      },
    },
    async (req) => sm.recognizeOnce(id(req)),
  );

  app.get(
    '/streams/:id/snapshot',
    { schema: { tags: ['streams'], params: IdParams, querystring: z.object({ overlay: BoolFlag, width: z.coerce.number().int().min(64).max(4096).optional() }) } },
    async (req, reply) => {
      const q = req.query as { overlay?: string; width?: number };
      const jpeg = await sm.snapshot(id(req), q.overlay === 'true' || q.overlay === '1', q.width);
      return reply.header('Content-Type', 'image/jpeg').header('Cache-Control', 'no-store').send(jpeg);
    },
  );
};
