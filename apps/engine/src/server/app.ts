// Fastify app: auth, error format, OpenAPI (spec §10.1).

import { readFileSync } from 'node:fs';
import Fastify, { type FastifyInstance, type FastifyRequest } from 'fastify';
import multipart from '@fastify/multipart';
import rateLimit from '@fastify/rate-limit';
import fastifyStatic from '@fastify/static';
import swagger from '@fastify/swagger';
import websocket from '@fastify/websocket';
import { hasZodFastifySchemaValidationErrors, jsonSchemaTransform, serializerCompiler, validatorCompiler, type ZodTypeProvider } from 'fastify-type-provider-zod';
import type { Engine } from '../engine.js';
import { ApiError } from '../util/errors.js';
import { adminRoutes } from './routes/admin.js';
import { eventRoutes } from './routes/events.js';
import { personRoutes } from './routes/persons.js';
import { recognizeRoutes } from './routes/recognize.js';
import { streamRoutes } from './routes/streams.js';
import { systemRoutes } from './routes/system.js';

export const API_PREFIX = '/api/v1';

declare module 'fastify' {
  interface FastifyRequest {
    actor: string;
  }
}

const LOOPBACK = new Set(['127.0.0.1', '::1', 'localhost']);

export interface ServerOptions {
  /** Directory with the built admin UI to serve at "/" (web mode without Electron). */
  uiDir?: string;
}

export async function buildServer(engine: Engine, opts: ServerOptions = {}): Promise<FastifyInstance> {
  const cfg = engine.cfg();
  if (!LOOPBACK.has(cfg.server.host) && !cfg.server.tls) {
    throw new Error('LAN access (server.host != 127.0.0.1) requires server.tls (spec §12.3)');
  }
  const maxBody = Math.ceil(cfg.enroll.maxPhotos * cfg.enroll.maxFileMb * 1024 * 1024 * 1.1) + 1024 * 1024;
  const app = Fastify({
    logger: false,
    bodyLimit: maxBody,
    https: cfg.server.tls ? { cert: readFileSync(cfg.server.tls.cert), key: readFileSync(cfg.server.tls.key) } : null,
  } as object).withTypeProvider<ZodTypeProvider>() as unknown as FastifyInstance;

  app.setValidatorCompiler(validatorCompiler);
  app.setSerializerCompiler(serializerCompiler);
  app.decorateRequest('actor', '');

  await app.register(swagger, {
    openapi: {
      info: { title: 'FaceID Engine API', version: '1.0.0', description: 'Local offline face recognition server' },
      servers: [{ url: API_PREFIX }],
      components: { securitySchemes: { bearer: { type: 'http', scheme: 'bearer' } } },
      security: [{ bearer: [] }],
    },
    transform: jsonSchemaTransform,
  });
  await app.register(multipart, {
    limits: { fileSize: cfg.enroll.maxFileMb * 1024 * 1024, files: cfg.enroll.maxPhotos, fields: 20, fieldSize: 256 * 1024 },
  });
  await app.register(rateLimit, {
    global: false,
    errorResponseBuilder: (_req, ctx) => new ApiError('RATE_LIMITED', `Rate limit exceeded, retry in ${ctx.after}`, {}, 429),
  });
  await app.register(websocket, {
    options: {
      maxPayload: 64 * 1024,
      // Token may be passed as WebSocket subprotocol "bearer.<token>" (browsers cannot set headers).
      handleProtocols: (protocols: Set<string>) => {
        for (const p of protocols) if (p.startsWith('bearer.')) return p;
        return false;
      },
    },
  });

  // CORS for explicitly allowed origins only (the Electron admin UI); preflights skip auth.
  const cors = new Set(cfg.server.corsOrigins);
  app.addHook('onRequest', async (req, reply) => {
    const origin = req.headers.origin;
    if (!origin || !cors.has(origin)) return;
    reply.header('Access-Control-Allow-Origin', origin).header('Vary', 'Origin').header('Access-Control-Expose-Headers', 'Content-Disposition');
    if (req.method === 'OPTIONS') {
      return reply
        .header('Access-Control-Allow-Methods', 'GET,POST,PATCH,DELETE,OPTIONS')
        .header('Access-Control-Allow-Headers', 'Authorization,Content-Type,If-Match')
        .header('Access-Control-Max-Age', '600')
        .status(204)
        .send();
    }
  });

  app.addHook('onRequest', async (req: FastifyRequest) => {
    const url = req.url.split('?')[0];
    if (!url.startsWith(API_PREFIX)) {
      if (opts.uiDir) return; // static admin UI files are public; the UI itself asks for the token
      throw new ApiError('NOT_FOUND', 'Not found');
    }
    if (url === `${API_PREFIX}/openapi.json`) return;
    if (url === `${API_PREFIX}/events` && req.headers.upgrade?.toLowerCase() === 'websocket') return; // WS authenticates itself
    const actor = engine.tokens.fromHeader(req.headers.authorization);
    if (!actor) throw new ApiError('UNAUTHORIZED', 'Missing or invalid bearer token');
    req.actor = actor;
  });

  app.setErrorHandler((err: any, req, reply) => {
    if (err instanceof ApiError) return reply.status(err.status).send(err.toBody());
    if (hasZodFastifySchemaValidationErrors(err)) {
      return reply.status(400).send(
        new ApiError('VALIDATION_ERROR', 'Request validation failed', {
          issues: err.validation.map((v: any) => ({ path: v.instancePath, message: v.message })),
        }).toBody(),
      );
    }
    if (err.code === 'FST_REQ_FILE_TOO_LARGE' || err.code === 'FST_ERR_CTP_BODY_TOO_LARGE') {
      return reply.status(413).send(new ApiError('FILE_TOO_LARGE', 'Payload too large').toBody());
    }
    if (err.code === 'FST_FILES_LIMIT' || err.code === 'FST_PARTS_LIMIT') {
      return reply.status(400).send(new ApiError('VALIDATION_ERROR', `Too many files (max ${engine.cfg().enroll.maxPhotos})`).toBody());
    }
    if (err.code === 'FST_INVALID_MULTIPART_CONTENT_TYPE' || err.code === 'FST_ERR_CTP_INVALID_MEDIA_TYPE') {
      return reply.status(415).send(new ApiError('VALIDATION_ERROR', 'Unsupported content type', {}, 415).toBody());
    }
    if (err.statusCode && err.statusCode < 500) {
      return reply.status(err.statusCode).send(new ApiError('VALIDATION_ERROR', err.message, {}, err.statusCode).toBody());
    }
    engine.log.error({ err: err.message, stack: err.stack, route: req.routeOptions?.url }, 'unhandled error');
    return reply.status(500).send(new ApiError('INTERNAL', 'Internal error').toBody());
  });
  app.setNotFoundHandler((_req, reply) => reply.status(404).send(new ApiError('NOT_FOUND', 'Route not found').toBody()));

  if (opts.uiDir) {
    await app.register(fastifyStatic, { root: opts.uiDir, prefix: '/', index: 'index.html' });
  }

  await app.register(
    async (api) => {
      await api.register(personRoutes(engine));
      await api.register(recognizeRoutes(engine));
      await api.register(streamRoutes(engine));
      await api.register(eventRoutes(engine));
      await api.register(systemRoutes(engine));
      await api.register(adminRoutes(engine));
      api.get('/openapi.json', { schema: { hide: true } }, async () => app.swagger());
    },
    { prefix: API_PREFIX },
  );
  return app;
}
