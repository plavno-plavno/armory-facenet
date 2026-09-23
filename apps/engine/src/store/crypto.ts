// AES-256-GCM file encryption (spec §12.2).
// Layout: "FENC" | version u8 = 1 | nonce(12) | tag(16) | ciphertext.

import { createCipheriv, createDecipheriv, randomBytes, scrypt as scryptCb } from 'node:crypto';
import { promisify } from 'node:util';

const MAGIC = Buffer.from('FENC');
const HEADER = MAGIC.length + 1 + 12 + 16;

const scrypt = promisify(scryptCb) as (pw: string, salt: Buffer, len: number, opts: object) => Promise<Buffer>;

export class DecryptError extends Error {}

export interface Cipher {
  readonly enabled: boolean;
  encrypt(plain: Uint8Array): Buffer;
  decrypt(data: Uint8Array): Buffer;
}

export class AesGcmCipher implements Cipher {
  readonly enabled = true;
  constructor(private readonly key: Buffer) {
    if (key.length !== 32) throw new Error('DEK must be 32 bytes');
  }

  encrypt(plain: Uint8Array): Buffer {
    const nonce = randomBytes(12);
    const c = createCipheriv('aes-256-gcm', this.key, nonce);
    const ct = Buffer.concat([c.update(plain), c.final()]);
    return Buffer.concat([MAGIC, Buffer.from([1]), nonce, c.getAuthTag(), ct]);
  }

  decrypt(data: Uint8Array): Buffer {
    const buf = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
    if (buf.length < HEADER || !buf.subarray(0, 4).equals(MAGIC) || buf[4] !== 1) {
      throw new DecryptError('Not an encrypted file');
    }
    const nonce = buf.subarray(5, 17);
    const tag = buf.subarray(17, 33);
    try {
      const d = createDecipheriv('aes-256-gcm', this.key, nonce);
      d.setAuthTag(tag);
      return Buffer.concat([d.update(buf.subarray(HEADER)), d.final()]);
    } catch {
      throw new DecryptError('Decryption failed (wrong key or corrupted file)');
    }
  }
}

/** Development-only pass-through (privacy.encryptAtRest = false). */
export class PlainCipher implements Cipher {
  readonly enabled = false;
  encrypt(plain: Uint8Array): Buffer {
    return Buffer.from(plain);
  }
  decrypt(data: Uint8Array): Buffer {
    return Buffer.from(data);
  }
}

export async function deriveKeyFromPassword(password: string, salt: Buffer): Promise<Buffer> {
  return scrypt(password, salt, 32, { N: 1 << 15, r: 8, p: 1, maxmem: 64 * 1024 * 1024 });
}
