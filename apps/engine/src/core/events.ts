// In-process event bus. Subscribers: WebSocket clients, webhooks, recognition journal.

import { EventEmitter } from 'node:events';
import type { EngineEvent } from '@faceid/shared';

export class EventBus {
  private readonly em = new EventEmitter();

  constructor() {
    this.em.setMaxListeners(0);
  }

  publish(e: EngineEvent): void {
    this.em.emit('event', e);
  }

  subscribe(fn: (e: EngineEvent) => void): () => void {
    this.em.on('event', fn);
    return () => this.em.off('event', fn);
  }
}
