// File-based storage of persons, photos and embeddings (spec §8.1, §8.4).

import { existsSync, promises as fs } from 'node:fs';
import path from 'node:path';
import type { Person } from '@faceid/shared';
import { writeFileAtomic } from './atomic.js';
import type { Cipher } from './crypto.js';
import { decodeEmbedding, encodeEmbedding } from './emb-format.js';

const PERSON_FILE = 'person.json.enc';

export interface PhotoFiles {
  originalJpeg: Buffer;
  alignedPng: Buffer;
  embeddings: Map<string, Float32Array>; // modelKey -> embedding
}

export class FileStore {
  readonly personsDir: string;
  readonly indexDir: string;

  constructor(
    readonly dataDir: string,
    private readonly cipher: Cipher,
  ) {
    this.personsDir = path.join(dataDir, 'persons');
    this.indexDir = path.join(dataDir, 'index');
  }

  personDir(id: string): string {
    if (!/^[0-9a-f-]{36}$/.test(id)) throw new Error('Invalid person id');
    return path.join(this.personsDir, id);
  }

  private photoPaths(personId: string, photoId: string) {
    const dir = this.personDir(personId);
    if (!/^[0-9a-f-]{36}$/.test(photoId)) throw new Error('Invalid photo id');
    return {
      original: path.join(dir, 'photos', `${photoId}.jpg.enc`),
      aligned: path.join(dir, 'faces', `${photoId}.png.enc`),
      embDir: path.join(dir, 'embeddings'),
    };
  }

  embeddingPath(personId: string, photoId: string, modelKey: string): string {
    return path.join(this.photoPaths(personId, photoId).embDir, `${photoId}.${modelKey}.emb.enc`);
  }

  private async writeEnc(file: string, plain: Uint8Array): Promise<void> {
    await writeFileAtomic(file, this.cipher.encrypt(plain));
  }

  private async readEnc(file: string): Promise<Buffer> {
    return this.cipher.decrypt(await fs.readFile(file));
  }

  async init(): Promise<{ removedDirs: string[]; removedFiles: number }> {
    await fs.mkdir(this.personsDir, { recursive: true });
    await fs.mkdir(this.indexDir, { recursive: true });
    const removedDirs: string[] = [];
    for (const name of await fs.readdir(this.personsDir)) {
      const dir = path.join(this.personsDir, name);
      // Leftovers of interrupted deletes, or creates that never reached person.json.
      if (name.startsWith('.trash-') || !existsSync(path.join(dir, PERSON_FILE))) {
        await fs.rm(dir, { recursive: true, force: true });
        removedDirs.push(name);
      }
    }
    return { removedDirs, removedFiles: 0 };
  }

  /** Remove photo/embedding files that are not referenced by person.json (interrupted adds). */
  async gcPerson(person: Person): Promise<number> {
    const known = new Set(person.photos.map((p) => p.id));
    let removed = 0;
    const dir = this.personDir(person.id);
    for (const sub of ['photos', 'faces', 'embeddings']) {
      const d = path.join(dir, sub);
      if (!existsSync(d)) continue;
      for (const f of await fs.readdir(d)) {
        const photoId = f.split('.')[0];
        if (f.endsWith('.tmp') || !known.has(photoId)) {
          await fs.rm(path.join(d, f), { force: true });
          removed++;
        }
      }
    }
    return removed;
  }

  async listPersonIds(): Promise<string[]> {
    if (!existsSync(this.personsDir)) return [];
    return (await fs.readdir(this.personsDir)).filter((n) => !n.startsWith('.'));
  }

  async readPerson(id: string): Promise<Person> {
    return JSON.parse((await this.readEnc(path.join(this.personDir(id), PERSON_FILE))).toString('utf8')) as Person;
  }

  async writePerson(p: Person): Promise<void> {
    await this.writeEnc(path.join(this.personDir(p.id), PERSON_FILE), Buffer.from(JSON.stringify(p), 'utf8'));
  }

  async writePhotoFiles(personId: string, photoId: string, files: PhotoFiles): Promise<void> {
    const p = this.photoPaths(personId, photoId);
    await this.writeEnc(p.original, files.originalJpeg);
    await this.writeEnc(p.aligned, files.alignedPng);
    for (const [key, emb] of files.embeddings) await this.writeEmbedding(personId, photoId, key, emb);
  }

  async writeEmbedding(personId: string, photoId: string, modelKey: string, emb: Float32Array): Promise<void> {
    await this.writeEnc(this.embeddingPath(personId, photoId, modelKey), encodeEmbedding(emb));
  }

  async readEmbedding(personId: string, photoId: string, modelKey: string): Promise<Float32Array> {
    return decodeEmbedding(await this.readEnc(this.embeddingPath(personId, photoId, modelKey)));
  }

  async readOriginal(personId: string, photoId: string): Promise<Buffer> {
    return this.readEnc(this.photoPaths(personId, photoId).original);
  }

  async readAligned(personId: string, photoId: string): Promise<Buffer> {
    return this.readEnc(this.photoPaths(personId, photoId).aligned);
  }

  async deletePhotoFiles(personId: string, photoId: string): Promise<void> {
    const p = this.photoPaths(personId, photoId);
    await fs.rm(p.original, { force: true });
    await fs.rm(p.aligned, { force: true });
    if (existsSync(p.embDir)) {
      for (const f of await fs.readdir(p.embDir)) if (f.startsWith(`${photoId}.`)) await fs.rm(path.join(p.embDir, f), { force: true });
    }
  }

  /** Rename to .trash-<id> first so a crash never leaves a half-deleted live person. */
  async deletePerson(personId: string): Promise<void> {
    const dir = this.personDir(personId);
    const trash = path.join(this.personsDir, `.trash-${personId}`);
    await fs.rename(dir, trash);
    await fs.rm(trash, { recursive: true, force: true });
  }

  // --- gallery index cache (spec §8.3) ---

  private indexPaths(modelKey: string) {
    const safe = modelKey.replace(/[^a-zA-Z0-9@._-]/g, '_');
    return {
      bin: path.join(this.indexDir, `gallery.${safe}.bin.enc`),
      json: path.join(this.indexDir, `gallery.${safe}.json.enc`),
    };
  }

  async writeIndexCache(modelKey: string, matrix: Float32Array, meta: unknown): Promise<void> {
    const p = this.indexPaths(modelKey);
    const bytes = Buffer.from(matrix.buffer, matrix.byteOffset, matrix.byteLength);
    // Force little-endian regardless of host: all supported platforms are LE, assert it.
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error('Big-endian hosts are not supported');
    await this.writeEnc(p.bin, bytes);
    await this.writeEnc(p.json, Buffer.from(JSON.stringify(meta)));
  }

  async readIndexCache(modelKey: string): Promise<{ matrix: Float32Array; meta: unknown } | null> {
    const p = this.indexPaths(modelKey);
    if (!existsSync(p.bin) || !existsSync(p.json)) return null;
    const meta = JSON.parse((await this.readEnc(p.json)).toString('utf8'));
    const bin = await this.readEnc(p.bin);
    const copy = new Uint8Array(bin.length);
    copy.set(bin);
    return { matrix: new Float32Array(copy.buffer), meta };
  }
}
