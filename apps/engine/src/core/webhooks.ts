// Webhook delivery (spec §10.5): HMAC-SHA256 signature, 3 s timeout, 3 retries with backoff.
// Failures are logged and never block the pipeline.

import { createHmac } from 'node:crypto';
import type { EngineEvent } from '@faceid/shared';
import type { EngineConfig } from '../config/schema.js';
import type { Logger } from '../util/logger.js';

export function signBody(secret: string, body: string): string {
  return `sha256=${createHmac('sha256', secret).update(body).digest('hex')}`;
}

export class WebhookDispatcher {
  constructor(
    private readonly cfg: () => EngineConfig,
    private readonly log: Logger,
    private readonly opts = { timeoutMs: 3000, retries: 3, backoffMs: 500 },
  ) {}

  handle(e: EngineEvent): void {
    for (const hook of this.cfg().webhooks) {
      if (!hook.events.includes(e.type)) continue;
      if (e.type === 'recognition.result' && hook.statuses && !hook.statuses.includes(e.status)) continue;
      void this.deliver(hook.url, hook.secret, e);
    }
  }

  async deliver(url: string, secret: string, e: EngineEvent): Promise<boolean> {
    const body = JSON.stringify(e);
    const headers = { 'Content-Type': 'application/json', 'X-Signature': signBody(secret, body), 'X-Event-Type': e.type };
    for (let attempt = 0; attempt <= this.opts.retries; attempt++) {
      if (attempt > 0) await new Promise((r) => setTimeout(r, this.opts.backoffMs * 2 ** (attempt - 1)));
      try {
        const res = await fetch(url, { method: 'POST', headers, body, signal: AbortSignal.timeout(this.opts.timeoutMs) });
        if (res.ok) return true;
        this.log.warn({ url: new URL(url).origin, status: res.status, attempt }, 'webhook delivery rejected');
      } catch (err) {
        this.log.warn({ url: new URL(url).origin, attempt, err: (err as Error).message }, 'webhook delivery failed');
      }
    }
    this.log.error({ url: new URL(url).origin, type: e.type }, 'webhook delivery gave up');
    return false;
  }
}
