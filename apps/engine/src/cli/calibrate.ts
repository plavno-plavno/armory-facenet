// CLI (M5): threshold calibration on a target-camera dataset (spec §15.3).
//
//   npm run calibrate -- --dataset <dir> --camera-profile <name> [--far 1e-3] [--embedder id] [--apply --url U --token T]
//
// Dataset layout:
//   <dir>/<personId>/enroll/*.jpg            reference photos, as used at real enrollment
//   <dir>/<personId>/passes/<pass>/*.jpg     frames of one pass in front of the target camera, or
//   <dir>/<personId>/passes/<pass>.mp4       a video of one pass
//   <dir>/groups.csv (optional)              personId,group   -> per-group error rates (demographic check)
//
// Output (in <dir>/calibration-<profile>/): report.md, scores.csv, curves.csv, profile.json (config patch).

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { EngineConfig } from '../config/schema.js';
import { rateAtThreshold, thresholdAtFar } from '../gallery/calibration.js';
import { GalleryIndex } from '../gallery/index.js';
import { meanEmbedding } from '../vision/embedder.js';
import { decodeImage } from '../vision/image.js';
import { loadVision, type VisionStack } from '../vision/index.js';
import { DefaultQualityAssessor } from '../vision/quality.js';
import type { Frame } from '../vision/types.js';

const root = path.resolve(import.meta.dirname, '../../../..');
const { values } = parseArgs({
  options: {
    dataset: { type: 'string' },
    'camera-profile': { type: 'string', default: 'default' },
    far: { type: 'string', default: '1e-3' },
    embedder: { type: 'string' },
    ep: { type: 'string', default: 'cpu' },
    models: { type: 'string', default: path.join(root, 'models') },
    topk: { type: 'string', default: '5' },
    apply: { type: 'boolean', default: false },
    url: { type: 'string', default: 'http://127.0.0.1:47810/api/v1' },
    token: { type: 'string' },
  },
});

const IMG = /\.(jpe?g|png|webp)$/i;
const VID = /\.(mp4|mkv|avi|mov|webm)$/i;

interface PassResult {
  personId: string;
  pass: string;
  frames: number;
  good: number;
  genuine: number | null; // probe vs own references
  impostorMax: number | null; // probe vs best other person (person treated as not enrolled)
  impostorPerson: string | null;
  margin: number | null; // genuine - impostorMax
  agreement: number | null;
  sharpness: number[]; // of the frames used for the probe, parallel to frameScores
  frameScores: number[];
}

/** Inverse standard normal CDF (Acklam's approximation). */
function normalQuantile(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const pl = 0.02425;
  if (p < pl) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - pl) return -normalQuantile(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function quantile(sorted: number[], q: number): number | null {
  if (!sorted.length) return null;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)))];
}

async function framesOfVideo(file: string): Promise<Frame[]> {
  const probe = spawnSync('ffprobe', ['-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', file]);
  const [w, h] = probe.stdout.toString().trim().split(',').map(Number);
  const r = spawnSync('ffmpeg', ['-v', 'error', '-i', file, '-vf', 'fps=10', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], { maxBuffer: 2 ** 31 - 1 });
  const size = w * h * 3;
  const out: Frame[] = [];
  for (let o = 0; o + size <= r.stdout.length; o += size) out.push({ data: new Uint8Array(r.stdout.subarray(o, o + size)), width: w, height: h, ts: out.length * 100, sourceId: 'cal' });
  return out;
}

async function loadPass(p: string): Promise<Frame[]> {
  if (statSync(p).isDirectory()) {
    const files = readdirSync(p).filter((f) => IMG.test(f)).sort();
    return Promise.all(files.map(async (f) => (await decodeImage(readFileSync(path.join(p, f)))).frame));
  }
  return framesOfVideo(p);
}

async function largestFace(v: VisionStack, frame: Frame) {
  const dets = await v.detector.detect(frame);
  if (!dets.length) return null;
  return dets.reduce((a, b) => (b.box[2] * b.box[3] > a.box[2] * a.box[3] ? b : a));
}

async function main() {
  const dataset = values.dataset;
  if (!dataset || !existsSync(dataset)) {
    console.error('usage: calibrate --dataset <dir> --camera-profile <name> [--far 1e-3]');
    process.exit(2);
  }
  const targetFar = Number(values.far);
  const cfg = EngineConfig.parse({});
  // Calibrate with the sharpness gate disabled; its recommended value is an output.
  const qa = new DefaultQualityAssessor({ ...cfg.quality, minSharpness: 0 });
  const v = await loadVision({ modelsDir: values.models!, embedderId: values.embedder, executionProvider: values.ep as 'cpu', detector: cfg.detector });
  const persons = readdirSync(dataset).filter((d) => existsSync(path.join(dataset, d, 'enroll')));
  console.error(`persons: ${persons.length}, model: ${v.info.embedder.modelKey}`);

  // 1. Gallery from enrollment photos.
  const g = new GalleryIndex(v.embedder.dim);
  for (const pid of persons) {
    for (const f of readdirSync(path.join(dataset, pid, 'enroll')).filter((x) => IMG.test(x))) {
      const { frame } = await decodeImage(readFileSync(path.join(dataset, pid, 'enroll', f)));
      const d = await largestFace(v, frame);
      if (!d) {
        console.error(`  no face: ${pid}/enroll/${f}`);
        continue;
      }
      g.upsertPhoto(pid, f, (await v.embedder.embed([v.aligner.align(frame, d.landmarks)]))[0]);
    }
  }

  // 2. Passes -> burst probes.
  const results: PassResult[] = [];
  const topK = Number(values.topk);
  for (const pid of persons) {
    const pdir = path.join(dataset, pid, 'passes');
    if (!existsSync(pdir)) continue;
    for (const pass of readdirSync(pdir).filter((x) => statSync(path.join(pdir, x)).isDirectory() || VID.test(x))) {
      const frames = await loadPass(path.join(pdir, pass));
      const cands: { emb?: Float32Array; aligned: ReturnType<VisionStack['aligner']['align']>; q: ReturnType<typeof qa.assess> }[] = [];
      for (const fr of frames) {
        const d = await largestFace(v, fr);
        if (!d) continue;
        const aligned = v.aligner.align(fr, d.landmarks);
        cands.push({ aligned, q: qa.assess(fr, d, aligned) });
      }
      const good = cands.filter((c) => c.q.passed).sort((a, b) => b.q.qualityScore - a.q.qualityScore);
      const best = good.slice(0, topK);
      const r: PassResult = { personId: pid, pass, frames: frames.length, good: good.length, genuine: null, impostorMax: null, impostorPerson: null, margin: null, agreement: null, sharpness: best.map((c) => c.q.sharpness), frameScores: [] };
      if (best.length) {
        const embs = await v.embedder.embed(best.map((c) => c.aligned));
        const probe = meanEmbedding(embs);
        r.genuine = g.scorePerson(pid, probe);
        const [imp] = g.match(probe, 1, { exclude: pid });
        r.impostorMax = imp?.score ?? null;
        r.impostorPerson = imp?.personId ?? null;
        if (r.genuine !== null && r.impostorMax !== null) r.margin = r.genuine - r.impostorMax;
        const all = g.match(probe, 1)[0]?.personId;
        r.agreement = embs.filter((e) => g.match(e, 1)[0]?.personId === all).length / embs.length;
        r.frameScores = embs.map((e) => g.scorePerson(pid, e) ?? 0);
      }
      results.push(r);
      process.stderr.write('.');
    }
  }
  process.stderr.write('\n');

  // 3. Statistics and recommendations.
  const scored = results.filter((r) => r.genuine !== null);
  const genuine = scored.map((r) => r.genuine!).sort((a, b) => a - b);
  const impostor = scored.filter((r) => r.impostorMax !== null).map((r) => r.impostorMax!).sort((a, b) => a - b);
  // Empirical quantile needs >= ~3/FAR impostor attempts; below that also use a Gaussian tail
  // estimate (mean + z·std) and take the stricter of the two.
  const empirical = thresholdAtFar(impostor, targetFar);
  const impMean = impostor.reduce((a, b) => a + b, 0) / Math.max(1, impostor.length);
  const impStd = Math.sqrt(impostor.reduce((a, b) => a + (b - impMean) ** 2, 0) / Math.max(1, impostor.length));
  const gaussian = impostor.length ? impMean + normalQuantile(1 - targetFar) * impStd : null;
  const accept = empirical === null ? null : impostor.length < 3 / targetFar ? Math.max(empirical, gaussian!) : empirical;
  const acceptR = accept === null ? cfg.match.acceptThreshold : Math.round(Math.max(accept, 0) * 1000) / 1000;
  const rejectQ = quantile(genuine, 0.01) ?? cfg.match.rejectThreshold;
  const reject = Math.round(Math.max(0, Math.min(rejectQ, acceptR - 0.05, (quantile(impostor, 0.999) ?? acceptR) + 0.02)) * 1000) / 1000;
  const margins = scored.filter((r) => r.margin !== null && r.genuine! >= acceptR).map((r) => r.margin!).sort((a, b) => a - b);
  const margin = Math.round(Math.min(0.1, Math.max(0.02, (quantile(margins, 0.01) ?? 0.05) * 0.5)) * 1000) / 1000;
  const goodSharp = scored.flatMap((r) => r.frameScores.map((s, i) => ({ s, sh: r.sharpness[i] }))).filter((x) => x.s >= acceptR).map((x) => x.sh).sort((a, b) => a - b);
  const minSharpness = Math.round(quantile(goodSharp, 0.05) ?? cfg.quality.minSharpness);
  const noImpostorCaveat = impostor.length < 1 / targetFar ? `⚠️ Only ${impostor.length} impostor attempts: FAR ${targetFar} cannot be verified; at least ${Math.ceil(3 / targetFar)} are recommended.` : '';

  // Decision simulation with the recommended thresholds (spec §7.7).
  const m = { acceptThreshold: acceptR, rejectThreshold: reject, margin, minFrameAgreement: cfg.match.minFrameAgreement };
  let tar = 0;
  let fm = 0;
  let unc = 0;
  for (const r of scored) {
    const s2 = r.impostorMax ?? -1;
    if (r.genuine! >= m.acceptThreshold && r.genuine! - s2 >= m.margin && (r.agreement ?? 0) >= m.minFrameAgreement) tar++;
    else if (r.genuine! >= m.rejectThreshold) unc++;
    // impostor scenario: person not enrolled -> top candidate is the best other person
    if (r.impostorMax !== null && r.impostorMax >= m.acceptThreshold) fm++;
  }

  // Per-group breakdown.
  const groupsFile = path.join(dataset, 'groups.csv');
  const groupOf = new Map<string, string>();
  if (existsSync(groupsFile)) for (const line of readFileSync(groupsFile, 'utf8').split('\n').slice(1)) {
    const [pid, grp] = line.split(',').map((s) => s?.trim());
    if (pid && grp) groupOf.set(pid, grp);
  }

  const outDir = path.join(dataset, `calibration-${values['camera-profile']}`);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(
    path.join(outDir, 'scores.csv'),
    ['personId,pass,frames,goodFrames,genuine,impostorMax,impostorPerson,margin,agreement', ...results.map((r) => [r.personId, r.pass, r.frames, r.good, r.genuine?.toFixed(4) ?? '', r.impostorMax?.toFixed(4) ?? '', r.impostorPerson ?? '', r.margin?.toFixed(4) ?? '', r.agreement?.toFixed(2) ?? ''].join(','))].join('\n'),
  );
  const curve: string[] = ['threshold,FAR,FRR'];
  for (let t = 0; t <= 0.8 + 1e-9; t += 0.01) curve.push(`${t.toFixed(2)},${rateAtThreshold(impostor, t, true).toFixed(5)},${rateAtThreshold(genuine, t, false).toFixed(5)}`);
  writeFileSync(path.join(outDir, 'curves.csv'), curve.join('\n'));
  const profile = { cameraProfile: values['camera-profile'], match: m, quality: { minSharpness } };
  writeFileSync(path.join(outDir, 'profile.json'), JSON.stringify(profile, null, 2));

  const pct = (x: number) => `${(100 * x).toFixed(2)} %`;
  const st = (a: number[]) => (a.length ? `n=${a.length}, mean ${(a.reduce((x, y) => x + y, 0) / a.length).toFixed(3)}, p1 ${quantile(a, 0.01)?.toFixed(3)}, p50 ${quantile(a, 0.5)?.toFixed(3)}, p99 ${quantile(a, 0.99)?.toFixed(3)}` : 'n=0');
  const md = [
    `# Calibration report — camera profile "${values['camera-profile']}"`,
    '',
    `Model: \`${v.info.embedder.modelKey}\`. Persons: ${persons.length}. Passes: ${results.length} (${results.length - scored.length} without a usable face). Target FAR per attempt: ${targetFar}.`,
    noImpostorCaveat,
    '',
    '## Score distributions (probe = mean of the burst)',
    '',
    `- genuine (probe vs own references): ${st(genuine)}`,
    `- impostor (probe vs best other person, as if not enrolled): ${st(impostor)}`,
    '',
    '## Recommended thresholds',
    '',
    '| Parameter | Value | Rule |',
    '|---|---|---|',
    `| match.acceptThreshold | **${m.acceptThreshold}** | smallest t with FAR ≤ ${targetFar} (empirical ${empirical?.toFixed(3)}, Gaussian tail ${gaussian?.toFixed(3)}) |`,
    `| match.rejectThreshold | **${m.rejectThreshold}** | ≈ 1st percentile of genuine, ≤ accept − 0.05 |`,
    `| match.margin | **${m.margin}** | half of the 1st percentile of genuine top1−top2 margins |`,
    `| quality.minSharpness | **${minSharpness}** | 5th percentile of sharpness on frames scoring ≥ accept |`,
    '',
    '## Simulated decisions with these thresholds',
    '',
    `- enrolled person → match on first attempt: ${pct(tar / scored.length)} (target ≥ 95 %, spec §15.4)`,
    `- enrolled person → uncertain (retry): ${pct(unc / scored.length)}`,
    `- not-enrolled person → false match: ${pct(fm / Math.max(1, impostor.length))}`,
    '',
    '## FAR / FRR',
    '',
    '| t | FAR | FRR |',
    '|---|---|---|',
    ...[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, m.acceptThreshold].sort((a, b) => a - b).map((t) => `| ${t} | ${pct(rateAtThreshold(impostor, t, true))} | ${pct(rateAtThreshold(genuine, t, false))} |`),
    '',
    ...(groupOf.size
      ? [
          '## Per-group error rates (demographic check, spec R5)',
          '',
          '| Group | passes | FRR @accept | FAR @accept |',
          '|---|---|---|---|',
          ...[...new Set(groupOf.values())].map((grp) => {
            const rs = scored.filter((r) => groupOf.get(r.personId) === grp);
            const gs = rs.map((r) => r.genuine!);
            const is = rs.filter((r) => r.impostorMax !== null).map((r) => r.impostorMax!);
            return `| ${grp} | ${rs.length} | ${pct(rateAtThreshold(gs, m.acceptThreshold, false))} | ${pct(rateAtThreshold(is, m.acceptThreshold, true))} |`;
          }),
          '',
        ]
      : []),
    `Apply: \`PATCH /api/v1/config\` with profile.json, or re-run with \`--apply --token <token>\`.`,
  ].join('\n');
  writeFileSync(path.join(outDir, 'report.md'), md);
  console.log(md);

  if (values.apply) {
    if (!values.token) throw new Error('--apply requires --token');
    const res = await fetch(`${values.url}/config`, { method: 'PATCH', headers: { Authorization: `Bearer ${values.token}`, 'Content-Type': 'application/json' }, body: JSON.stringify(profile) });
    console.error(`apply: HTTP ${res.status}`);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
