// Audit log (spec §12.4): identifiers only, never names or biometrics.

import { promises as fs } from 'node:fs';
import path from 'node:path';

export interface AuditEntry {
  actor: string; // token id
  action: string;
  personId?: string;
  photoId?: string;
  flags?: string[];
  details?: Record<string, string | number | boolean>;
}

export class AuditLog {
  private readonly dir: string;
  private chain: Promise<void> = Promise.resolve();

  constructor(dataDir: string) {
    this.dir = path.join(dataDir, 'logs', 'audit');
  }

  append(e: AuditEntry): Promise<void> {
    const ts = new Date().toISOString();
    const line = JSON.stringify({ ts, ...e }) + '\n';
    const file = path.join(this.dir, `${ts.slice(0, 10)}.jsonl`);
    this.chain = this.chain.then(async () => {
      await fs.mkdir(this.dir, { recursive: true });
      await fs.appendFile(file, line);
    });
    return this.chain;
  }
}
