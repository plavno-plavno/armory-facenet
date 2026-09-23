// Admin: reindex, jobs, calibration stats, token rotation (spec §10.6).

import { Job } from '@faceid/shared';
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { Engine } from '../../engine.js';
import { galleryScoreDistributions } from '../../gallery/calibration.js';
import { ApiError } from '../../util/errors.js';
import { errors } from './common.js';

export const adminRoutes = (engine: Engine) => async (app: FastifyInstance) => {
  app.post('/admin/reindex', { schema: { tags: ['admin'], querystring: z.object({ onlyMissing: z.enum(['true', 'false']).optional() }) } }, async (req, reply) => {
    const jobId = engine.startReindex((req.query as { onlyMissing?: string }).onlyMissing === 'true');
    await engine.audit.append({ actor: req.actor, action: 'admin.reindex', details: { jobId } });
    return reply.status(202).send({ jobId });
  });

  app.get('/admin/jobs/:id', { schema: { tags: ['admin'], params: z.object({ id: z.string() }), response: { 200: Job, ...errors(404) } } }, async (req) => {
    const job = engine.jobs.get((req.params as { id: string }).id);
    if (!job) throw new ApiError('NOT_FOUND', 'Job not found');
    return job;
  });

  app.get(
    '/admin/calibration/gallery-scores',
    { schema: { tags: ['admin'], querystring: z.object({ bins: z.coerce.number().int().min(10).max(200).default(50), maxImpostorPairs: z.coerce.number().int().min(100).max(5_000_000).default(200_000) }) } },
    async (req) => {
      const q = req.query as { bins: number; maxImpostorPairs: number };
      return galleryScoreDistributions(engine.index, q.bins, q.maxImpostorPairs, engine.cfg().match);
    },
  );

  app.post('/admin/token/rotate', { schema: { tags: ['admin'] } }, async (req) => {
    const token = await engine.tokens.rotate();
    await engine.audit.append({ actor: req.actor, action: 'token.rotate' });
    return { token };
  });
};
