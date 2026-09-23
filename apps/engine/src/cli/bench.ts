// CLI (M1): CPU/GPU benchmark -> docs/benchmark.md
//   npm run bench -- [--ep cpu,cuda] [--embedders lvface-t-glint360k,lvface-s-glint360k] [--out docs/benchmark.md]

import { readFileSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { EngineConfig } from '../config/schema.js';
import { GalleryIndex } from '../gallery/index.js';
import { l2normalize } from '../vision/embedder.js';
import { decodeImage } from '../vision/image.js';
import { loadVision } from '../vision/index.js';
import { loadManifest } from '../vision/models.js';
import { resizeBilinear } from '../vision/preprocess.js';
import { DefaultQualityAssessor } from '../vision/quality.js';
import type { Frame } from '../vision/types.js';

const root = path.resolve(import.meta.dirname, '../../../..');
const { values } = parseArgs({
  options: {
    ep: { type: 'string', default: 'cpu' },
    embedders: { type: 'string' },
    models: { type: 'string', default: path.join(root, 'models') },
    out: { type: 'string', default: path.join(root, 'docs/benchmark.md') },
    iters: { type: 'string', default: '30' },
  },
});

async function time(n: number, fn: () => Promise<unknown> | unknown): Promise<number> {
  for (let i = 0; i < 3; i++) await fn(); // warm-up
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await fn();
  return (performance.now() - t0) / n;
}

const f1 = (v: number) => v.toFixed(1);

async function main() {
  const N = Number(values.iters);
  const cfg = EngineConfig.parse({});
  const manifest = loadManifest(values.models!);
  const allEmb = [manifest.embedder, ...(manifest.embedders ?? [])];
  const embIds = values.embedders ? values.embedders.split(',') : allEmb.map((e) => e.id);
  const eps = values.ep!.split(',');

  // Test frame: a group photo scaled into 1280x720 and 1920x1080.
  const { frame: src } = await decodeImage(readFileSync(path.join(root, 'tests/fixtures/raw/t1.jpg')));
  const frameOf = (w: number, h: number): Frame => ({ data: resizeBilinear(src.data, src.width, src.height, w, h), width: w, height: h, ts: 0, sourceId: 'bench' });
  const f720 = frameOf(1280, 720);
  const f1080 = frameOf(1920, 1080);

  const lines: string[] = [];
  const cpu = os.cpus();
  lines.push('# Benchmark', '', `Date: ${new Date().toISOString()}`, '');
  lines.push(`- CPU: ${cpu[0]?.model} (${cpu.length} logical cores)`, `- RAM: ${Math.round(os.totalmem() / 2 ** 30)} GiB`, `- OS: ${os.type()} ${os.release()} ${os.arch()}`, `- Node: ${process.version}`, '');

  const base = await loadVision({ modelsDir: values.models!, executionProvider: 'cpu', detector: cfg.detector });
  const dets720 = await base.detector.detect(f720);
  const det720 = await time(N, () => base.detector.detect(f720));
  const det1080 = await time(N, () => base.detector.detect(f1080));
  const align = await time(N * 10, () => base.aligner.align(f720, dets720[0].landmarks));
  const aligned = base.aligner.align(f720, dets720[0].landmarks);
  const qa = new DefaultQualityAssessor(cfg.quality);
  const quality = await time(N * 10, () => qa.assess(f720, dets720[0], aligned));

  lines.push('## Detection (YuNet, CPU, inputLongSide=640)', '', '| Frame | ms/frame | max fps (1 stream) |', '|---|---|---|');
  lines.push(`| 1280×720 (${dets720.length} faces) | ${f1(det720)} | ${f1(1000 / det720)} |`, `| 1920×1080 | ${f1(det1080)} | ${f1(1000 / det1080)} |`, '');
  lines.push(`Alignment: ${align.toFixed(2)} ms/face, quality metrics: ${quality.toFixed(2)} ms/face.`, '');

  lines.push('## Embeddings (LVFace)', '', '| Model | EP | batch=1 ms | batch=5 ms | ms/face @5 |', '|---|---|---|---|---|');
  const embResults: Record<string, number> = {};
  for (const id of embIds) {
    for (const ep of eps) {
      try {
        const v = await loadVision({ modelsDir: values.models!, embedderId: id, executionProvider: ep as 'cpu', detector: cfg.detector, verifyChecksums: false });
        if (v.info.embedder.executionProvider !== ep) throw new Error(`fell back to ${v.info.embedder.executionProvider}`);
        const b1 = await time(N, () => v.embedder.embed([aligned]));
        const b5 = await time(N, () => v.embedder.embed([aligned, aligned, aligned, aligned, aligned]));
        embResults[`${id}/${ep}`] = b5;
        lines.push(`| ${id} | ${ep} | ${f1(b1)} | ${f1(b5)} | ${f1(b5 / 5)} |`);
      } catch (e) {
        lines.push(`| ${id} | ${ep} | n/a | n/a | ${(e as Error).message.split('\n')[0].slice(0, 80)} |`);
      }
    }
  }
  lines.push('');

  lines.push('## Matching (pure TS, brute force)', '', '| Gallery | ms/probe |', '|---|---|');
  for (const n of [10_000, 50_000]) {
    const g = new GalleryIndex(512, n);
    const e = l2normalize(Float32Array.from({ length: 512 }, () => Math.random() - 0.5));
    for (let i = 0; i < n; i++) g.upsertPhoto(`p${i % (n / 5)}`, `ph${i}`, e);
    lines.push(`| ${n.toLocaleString('en')} × 512 | ${f1(await time(20, () => g.match(e, 2)))} |`);
  }
  lines.push('');

  // Latency model (spec §13): the burst closes after maxFrames at detectFps (or windowMs),
  // then topK embeddings + matching. Collection dominates.
  const collect = Math.min(((cfg.burst.maxFrames - 1) * 1000) / cfg.pipeline.detectFps, cfg.burst.windowMs);
  lines.push('## Estimated recognition latency (first detection → event)', '');
  lines.push(`Burst collection: ${collect} ms (maxFrames=${cfg.burst.maxFrames} @ ${cfg.pipeline.detectFps} fps, windowMs=${cfg.burst.windowMs}).`, '');
  lines.push('| Embedder/EP | + detect per frame | + embed topK=5 | total estimate | spec target |', '|---|---|---|---|---|');
  for (const [k, b5] of Object.entries(embResults)) {
    const total = collect + det720 + b5 + 25;
    lines.push(`| ${k} | ${f1(det720)} | ${f1(b5)} | ${Math.round(total)} ms | ≤ 1500 ms ${total <= 1500 ? '✅' : '❌'} |`);
  }
  lines.push('', `Engine RSS after loading models: ${Math.round(process.memoryUsage().rss / 2 ** 20)} MiB.`, '');

  const md = lines.join('\n');
  writeFileSync(values.out!, md);
  console.log(md);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
