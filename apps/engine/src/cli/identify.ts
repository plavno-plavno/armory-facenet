// CLI (M1): identify faces in an image.
//   npm run identify -- <image> --gallery <dir>        folder-per-person gallery built on the fly
//   npm run identify -- <image> --data-dir <dir> [--password <pw>]   the engine's stored gallery (read-only)

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import pino from 'pino';
import type { Person } from '@faceid/shared';
import { EngineConfig } from '../config/schema.js';
import { GalleryIndex } from '../gallery/index.js';
import { IndexManager } from '../gallery/rebuild.js';
import { decide } from '../pipeline/decision.js';
import { AesGcmCipher, PlainCipher } from '../store/crypto.js';
import { FileStore } from '../store/file-store.js';
import { loadOrCreateDek, passwordWrapper, readKeystoreMode } from '../store/keystore.js';
import { decodeImage } from '../vision/image.js';
import { loadVision } from '../vision/index.js';
import { DefaultQualityAssessor } from '../vision/quality.js';

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    gallery: { type: 'string' },
    'data-dir': { type: 'string' },
    password: { type: 'string' },
    models: { type: 'string', default: path.resolve(import.meta.dirname, '../../../../models') },
    embedder: { type: 'string' },
    ep: { type: 'string', default: 'cpu' },
    topk: { type: 'string', default: '3' },
  },
});

async function main() {
  const image = positionals[0];
  if (!image || (!values.gallery && !values['data-dir'])) {
    console.error('usage: identify <image> (--gallery <dir> | --data-dir <dir> [--password <pw>]) [--embedder id] [--ep cpu|cuda|auto]');
    process.exit(2);
  }
  const cfg = EngineConfig.parse({});
  const vision = await loadVision({ modelsDir: values.models!, embedderId: values.embedder, executionProvider: values.ep as 'cpu', detector: cfg.detector });
  const index = new GalleryIndex(vision.embedder.dim);
  const names = new Map<string, string>();

  const embedFile = async (file: string) => {
    const { frame } = await decodeImage(readFileSync(file));
    const dets = await vision.detector.detect(frame);
    if (dets.length === 0) return null;
    const d = dets.reduce((a, b) => (b.box[2] * b.box[3] > a.box[2] * a.box[3] ? b : a));
    return (await vision.embedder.embed([vision.aligner.align(frame, d.landmarks)]))[0];
  };

  if (values.gallery) {
    for (const person of readdirSync(values.gallery)) {
      const dir = path.join(values.gallery, person);
      if (!statSync(dir).isDirectory()) continue;
      names.set(person, person);
      for (const f of readdirSync(dir)) {
        const e = await embedFile(path.join(dir, f));
        if (e) index.upsertPhoto(person, f, e);
      }
    }
  } else {
    const dataDir = values['data-dir']!;
    const mode = await readKeystoreMode(dataDir);
    let cipher;
    if (mode === 'password') cipher = new AesGcmCipher(await loadOrCreateDek(dataDir, passwordWrapper(values.password ?? '')));
    else if (mode === null) cipher = new PlainCipher();
    else throw new Error('This data dir is protected by Electron safeStorage; use the API instead.');
    const store = new FileStore(dataDir, cipher);
    const persons: Person[] = [];
    for (const id of await store.listPersonIds()) persons.push(await store.readPerson(id));
    persons.forEach((p) => names.set(p.id, `${p.lastName} ${p.firstName}`));
    await new IndexManager(index, vision.embedder.modelKey, store, pino({ level: 'silent' })).load(persons);
  }

  const { frame } = await decodeImage(readFileSync(image));
  const dets = await vision.detector.detect(frame);
  const aligned = dets.map((d) => vision.aligner.align(frame, d.landmarks));
  const embs = await vision.embedder.embed(aligned);
  const qa = new DefaultQualityAssessor(cfg.quality);
  const faces = dets.map((d, i) => {
    const cands = index.match(embs[i], Math.max(2, Number(values.topk)));
    const q = qa.assess(frame, d, aligned[i]);
    return {
      box: d.box.map(Math.round),
      detScore: +d.score.toFixed(3),
      quality: { passed: q.passed, reasons: q.reasons },
      status: q.passed ? decide(cands[0]?.score ?? null, cands[1]?.score ?? null, 1, cfg.match) : 'low_quality',
      candidates: cands.slice(0, Number(values.topk)).map((c) => ({ person: names.get(c.personId) ?? c.personId, score: +c.score.toFixed(4) })),
    };
  });
  console.log(JSON.stringify({ image, model: vision.info.embedder, gallery: index.stats(), faces }, null, 2));
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
