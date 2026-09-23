// Atomic file writes: *.tmp -> fsync -> rename -> fsync(dir) (spec §8.4).

import { randomBytes } from 'node:crypto';
import { promises as fs } from 'node:fs';
import path from 'node:path';

async function fsyncDir(dir: string): Promise<void> {
  if (process.platform === 'win32') return; // directories cannot be opened for fsync on Windows
  const h = await fs.open(dir, 'r');
  try {
    await h.sync();
  } finally {
    await h.close();
  }
}

export async function writeFileAtomic(file: string, data: Uint8Array): Promise<void> {
  const dir = path.dirname(file);
  await fs.mkdir(dir, { recursive: true });
  const tmp = `${file}.${randomBytes(4).toString('hex')}.tmp`;
  const h = await fs.open(tmp, 'w');
  try {
    await h.writeFile(data);
    await h.sync();
  } finally {
    await h.close();
  }
  try {
    await fs.rename(tmp, file);
  } catch (e) {
    await fs.rm(tmp, { force: true });
    throw e;
  }
  await fsyncDir(dir);
}
