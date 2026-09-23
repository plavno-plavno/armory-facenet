// API token (spec §12.3): 32 random bytes base64url, stored encrypted with the DEK
// (itself wrapped by safeStorage), rotated via /admin/token/rotate.

import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { existsSync, promises as fs } from 'node:fs';
import path from 'node:path';
import { writeFileAtomic } from '../store/atomic.js';
import type { Cipher } from '../store/crypto.js';

export function tokenId(token: string): string {
  return `tok_${createHash('sha256').update(token).digest('hex').slice(0, 12)}`;
}

export class TokenStore {
  private token = '';
  private readonly file: string;

  constructor(
    dataDir: string,
    private readonly cipher: Cipher,
  ) {
    this.file = path.join(dataDir, 'token.enc');
  }

  async init(): Promise<void> {
    if (existsSync(this.file)) this.token = this.cipher.decrypt(await fs.readFile(this.file)).toString('utf8');
    else await this.rotate();
  }

  current(): string {
    return this.token;
  }

  async rotate(): Promise<string> {
    const t = randomBytes(32).toString('base64url');
    await writeFileAtomic(this.file, this.cipher.encrypt(Buffer.from(t)));
    this.token = t;
    return t;
  }

  /** Constant-time check; returns the token id for audit or null. */
  verify(candidate: string | undefined): string | null {
    if (!candidate || !this.token) return null;
    const a = Buffer.from(candidate);
    const b = Buffer.from(this.token);
    if (a.length !== b.length || !timingSafeEqual(a, b)) return null;
    return tokenId(candidate);
  }

  fromHeader(h: string | undefined): string | null {
    const m = /^Bearer\s+(.+)$/i.exec(h ?? '');
    return this.verify(m?.[1]);
  }
}
