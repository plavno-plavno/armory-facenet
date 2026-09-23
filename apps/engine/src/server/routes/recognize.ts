// One-shot recognition (spec §10.4).

import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { Engine } from '../../engine.js';
import { ApiError } from '../../util/errors.js';
import { readMultipart } from '../multipart.js';
import { errors } from './common.js';

export const recognizeRoutes = (engine: Engine) => async (app: FastifyInstance) => {
  app.post(
    '/recognize/identify',
    {
      config: { rateLimit: { max: 20, timeWindow: 1000 } },
      schema: {
        tags: ['recognize'],
        summary: 'Identify all faces in a photo (multipart: photo, topK)',
        consumes: ['multipart/form-data'],
        querystring: z.object({ topK: z.coerce.number().int().min(1).max(10).optional() }),
        response: errors(400, 413, 415),
      },
    },
    async (req) => {
      const mp = await readMultipart(req, ['photo']);
      if (mp.files.length !== 1) throw new ApiError('VALIDATION_ERROR', 'Exactly one "photo" file is required');
      const topK = Math.min(10, Math.max(1, Number(mp.fields.topK ?? (req.query as { topK?: number }).topK ?? 3) || 3));
      return engine.recognize.identify(mp.files[0], topK);
    },
  );

  app.post(
    '/recognize/verify',
    {
      config: { rateLimit: { max: 20, timeWindow: 1000 } },
      schema: {
        tags: ['recognize'],
        summary: 'Verify a photo against a person (multipart: photo, personId)',
        consumes: ['multipart/form-data'],
        response: { 200: z.object({ score: z.number(), match: z.boolean(), threshold: z.number() }), ...errors(400, 404, 413, 415, 422) },
      },
    },
    async (req) => {
      const mp = await readMultipart(req, ['photo']);
      if (mp.files.length !== 1) throw new ApiError('VALIDATION_ERROR', 'Exactly one "photo" file is required');
      const personId = mp.fields.personId ?? (req.query as { personId?: string }).personId;
      if (!personId) throw new ApiError('VALIDATION_ERROR', 'personId is required');
      engine.persons.get(personId);
      return engine.recognize.verify(mp.files[0], personId);
    },
  );
};
