// Health and config (spec §10.6).

import type { FastifyInstance } from 'fastify';
import { z } from 'zod';
import type { Engine } from '../../engine.js';
import { ApiError } from '../../util/errors.js';

/** Webhook secrets are write-only through the API. */
function publicConfig(engine: Engine) {
  const c = engine.cfg();
  return { ...c, webhooks: c.webhooks.map((w) => ({ ...w, secret: '***' })) };
}

export const systemRoutes = (engine: Engine) => async (app: FastifyInstance) => {
  app.get('/health', { schema: { tags: ['system'] } }, async () => engine.health());

  app.get('/config', { schema: { tags: ['system'] } }, async () => ({ config: publicConfig(engine), restartRequired: engine.cfgStore.pendingRestart() }));

  app.patch('/config', { schema: { tags: ['system'], body: z.record(z.string(), z.unknown()) } }, async (req) => {
    const patch = req.body as Record<string, any>;
    // Keep existing secrets when the client echoes the masked value back.
    if (Array.isArray(patch.webhooks)) {
      const cur = engine.cfg().webhooks;
      patch.webhooks = patch.webhooks.map((w: any) => (w?.secret === '***' ? { ...w, secret: cur.find((c) => c.url === w.url)?.secret } : w));
    }
    try {
      const r = await engine.cfgStore.patch(patch);
      await engine.audit.append({ actor: req.actor, action: 'config.update', details: { keys: r.changed.join(',') } });
      return { config: publicConfig(engine), changed: r.changed, restartRequired: r.restartRequired };
    } catch (e) {
      if (e instanceof z.ZodError) {
        throw new ApiError('VALIDATION_ERROR', 'Invalid configuration', { issues: e.issues.map((i) => ({ path: i.path.join('.'), message: i.message })) });
      }
      throw e;
    }
  });
};
