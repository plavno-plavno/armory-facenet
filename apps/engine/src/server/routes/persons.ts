// Persons CRUD (spec §10.3).

import { Person, PersonCreate, PersonList, PersonPatch, PersonStatus, Photo } from '@faceid/shared';
import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { Engine } from '../../engine.js';
import { ApiError } from '../../util/errors.js';
import { parseBool, parseFaceBox, parseJsonField, parsePhotoSources, readMultipart } from '../multipart.js';
import { BoolFlag, errors, IdParams } from './common.js';

const EnrollQuery = z.object({ allowDuplicate: BoolFlag, force: BoolFlag, faceBox: z.string().optional() });
export const personRoutes = (engine: Engine) => async (app: FastifyInstance) => {
  const svc = engine.persons;
  const enrollRate = { rateLimit: { max: engine.cfg().server.enrollRatePerSec, timeWindow: 1000 } };

  app.post(
    '/persons',
    {
      config: enrollRate,
      schema: {
        tags: ['persons'],
        summary: 'Create a person with 1..N photos (multipart: data=JSON, photos=files, optional photoSources=JSON ["upload"|"camera"])',
        consumes: ['multipart/form-data'],
        querystring: EnrollQuery,
        response: { 201: Person, ...errors(400, 409, 413, 415, 422) },
      },
    },
    async (req, reply) => {
      const q = req.query as z.infer<typeof EnrollQuery>;
      const mp = await readMultipart(req, ['photos', 'photo']);
      const parsed = PersonCreate.safeParse(parseJsonField(mp.fields.data, 'data') ?? {});
      if (!parsed.success) {
        throw new ApiError('VALIDATION_ERROR', 'Invalid person data', { issues: parsed.error.issues.map((i) => ({ path: i.path.join('.'), message: i.message })) });
      }
      const person = await svc.create(
        parsed.data,
        mp.files,
        {
          allowDuplicate: parseBool(q.allowDuplicate ?? mp.fields.allowDuplicate),
          force: parseBool(q.force ?? mp.fields.force),
          faceBox: parseFaceBox(q.faceBox ?? mp.fields.faceBox),
          sources: parsePhotoSources(mp.fields.photoSources, mp.files.length),
        },
        req.actor,
      );
      return reply.status(201).send(person);
    },
  );

  app.get(
    '/persons',
    {
      schema: {
        tags: ['persons'],
        querystring: z.object({
          q: z.string().max(200).optional(),
          status: PersonStatus.optional(),
          limit: z.coerce.number().int().min(1).max(100).default(50),
          cursor: z.string().optional(),
        }),
        response: { 200: PersonList },
      },
    },
    async (req) => svc.list(req.query as { q?: string; status?: string; limit: number; cursor?: string }),
  );

  app.get('/persons/:id', { schema: { tags: ['persons'], params: IdParams, response: { 200: Person, ...errors(404) } } }, async (req) =>
    svc.get((req.params as { id: string }).id),
  );

  app.patch(
    '/persons/:id',
    {
      schema: {
        tags: ['persons'],
        params: IdParams,
        headers: z.object({ 'if-match': z.string().regex(/^"?\d+"?$/).optional() }).passthrough(),
        body: PersonPatch,
        response: { 200: Person, ...errors(400, 404, 409, 412) },
      },
    },
    async (req) => {
      const ifMatch = req.headers['if-match'];
      const version = ifMatch ? Number(String(ifMatch).replace(/"/g, '')) : undefined;
      return svc.patch((req.params as { id: string }).id, req.body as PersonPatch, version, req.actor);
    },
  );

  app.delete(
    '/persons/:id',
    { schema: { tags: ['persons'], params: IdParams, querystring: z.object({ purgeEvents: BoolFlag }) } },
    async (req, reply) => {
      await svc.delete((req.params as { id: string }).id, parseBool((req.query as { purgeEvents?: string }).purgeEvents), req.actor);
      return reply.status(204).send();
    },
  );

  app.get('/persons/:id/export', { schema: { tags: ['persons'], params: IdParams } }, async (req, reply) => {
    const id = (req.params as { id: string }).id;
    const zip = await svc.exportZip(id, req.actor);
    return reply.header('Content-Type', 'application/zip').header('Content-Disposition', `attachment; filename="person-${id}.zip"`).send(zip);
  });

  app.get('/persons/:id/photos', { schema: { tags: ['photos'], params: IdParams, response: { 200: z.array(Photo), ...errors(404) } } }, async (req) =>
    svc.get((req.params as { id: string }).id).photos,
  );

  app.post(
    '/persons/:id/photos',
    {
      config: enrollRate,
      schema: {
        tags: ['photos'],
        summary: 'Add photos (multipart: photos=files)',
        consumes: ['multipart/form-data'],
        params: IdParams,
        querystring: EnrollQuery,
        response: { 201: z.array(Photo), ...errors(400, 404, 409, 413, 415, 422) },
      },
    },
    async (req, reply) => {
      const q = req.query as z.infer<typeof EnrollQuery>;
      const mp = await readMultipart(req, ['photos', 'photo']);
      const photos = await svc.addPhotos(
        (req.params as { id: string }).id,
        mp.files,
        {
          force: parseBool(q.force ?? mp.fields.force),
          allowDuplicate: parseBool(q.allowDuplicate ?? mp.fields.allowDuplicate),
          faceBox: parseFaceBox(q.faceBox ?? mp.fields.faceBox),
          sources: parsePhotoSources(mp.fields.photoSources, mp.files.length),
        },
        req.actor,
      );
      return reply.status(201).send(photos);
    },
  );

  app.post(
    '/persons/:id/photos/capture',
    {
      config: enrollRate,
      schema: {
        tags: ['photos'],
        summary: 'Enroll from a running camera stream (spec §9.3)',
        params: IdParams,
        querystring: z.object({ force: BoolFlag, allowDuplicate: BoolFlag }),
        body: z.object({ sourceId: z.string() }),
        response: { 201: z.array(Photo), ...errors(400, 404, 409, 422) },
      },
    },
    async (req, reply) => {
      const id = (req.params as { id: string }).id;
      svc.get(id);
      const q = req.query as { force?: string; allowDuplicate?: string };
      const analyzed = await engine.streams.capture((req.body as { sourceId: string }).sourceId);
      const photos = await svc.addAnalyzed(id, analyzed, 'camera', { force: parseBool(q.force), allowDuplicate: parseBool(q.allowDuplicate) }, req.actor);
      return reply.status(201).send(photos);
    },
  );

  app.get(
    '/persons/:id/photos/:photoId',
    {
      schema: {
        tags: ['photos'],
        params: z.object({ id: z.string().uuid(), photoId: z.string().uuid() }),
        querystring: z.object({ variant: z.enum(['original', 'aligned']).default('original') }),
      },
    },
    async (req, reply) => {
      const p = req.params as { id: string; photoId: string };
      const img = await svc.photoImage(p.id, p.photoId, (req.query as { variant: 'original' | 'aligned' }).variant);
      return reply.header('Content-Type', img.type).header('Cache-Control', 'no-store').send(img.data);
    },
  );

  app.delete(
    '/persons/:id/photos/:photoId',
    { schema: { tags: ['photos'], params: z.object({ id: z.string().uuid(), photoId: z.string().uuid() }) } },
    async (req, reply) => {
      const p = req.params as { id: string; photoId: string };
      await svc.deletePhoto(p.id, p.photoId, req.actor);
      return reply.status(204).send();
    },
  );
};
