// keystore.bin: the data-encryption key (DEK) wrapped by a platform secret (spec §12.2).
// In Electron the wrapper is safeStorage (main process); standalone / Linux without a
// secret service use a password (scrypt). The engine only ever sees the unwrapped DEK.

import { randomBytes } from 'node:crypto';
import { existsSync, promises as fs } from 'node:fs';
import path from 'node:path';
import { AesGcmCipher, deriveKeyFromPassword } from './crypto.js';
import { writeFileAtomic } from './atomic.js';

export type KeystoreMode = 'safeStorage' | 'password';

interface KeystoreFile {
  version: 1;
  mode: KeystoreMode;
  salt?: string; // base64, password mode
  wrapped: string; // base64
}

export interface KeyWrapper {
  mode: KeystoreMode;
  wrap(dek: Buffer): Promise<{ wrapped: Buffer; salt?: Buffer }>;
  unwrap(wrapped: Buffer, salt?: Buffer): Promise<Buffer>;
}

export function passwordWrapper(password: string): KeyWrapper {
  return {
    mode: 'password',
    async wrap(dek) {
      const salt = randomBytes(16);
      const kek = await deriveKeyFromPassword(password, salt);
      return { wrapped: new AesGcmCipher(kek).encrypt(dek), salt };
    },
    async unwrap(wrapped, salt) {
      if (!salt) throw new Error('keystore: missing salt');
      const kek = await deriveKeyFromPassword(password, salt);
      return new AesGcmCipher(kek).decrypt(wrapped);
    },
  };
}

export function keystorePath(dataDir: string): string {
  return path.join(dataDir, 'keystore.bin');
}

export async function readKeystoreMode(dataDir: string): Promise<KeystoreMode | null> {
  const p = keystorePath(dataDir);
  if (!existsSync(p)) return null;
  return (JSON.parse(await fs.readFile(p, 'utf8')) as KeystoreFile).mode;
}

/** Load the DEK, creating and persisting a new one on first run. */
export async function loadOrCreateDek(dataDir: string, wrapper: KeyWrapper): Promise<Buffer> {
  const p = keystorePath(dataDir);
  if (existsSync(p)) {
    const ks = JSON.parse(await fs.readFile(p, 'utf8')) as KeystoreFile;
    if (ks.mode !== wrapper.mode) throw new Error(`keystore was created in "${ks.mode}" mode, not "${wrapper.mode}"`);
    return wrapper.unwrap(Buffer.from(ks.wrapped, 'base64'), ks.salt ? Buffer.from(ks.salt, 'base64') : undefined);
  }
  const dek = randomBytes(32);
  const { wrapped, salt } = await wrapper.wrap(dek);
  const file: KeystoreFile = { version: 1, mode: wrapper.mode, wrapped: wrapped.toString('base64') };
  if (salt) file.salt = salt.toString('base64');
  await writeFileAtomic(p, Buffer.from(JSON.stringify(file)));
  return dek;
}
