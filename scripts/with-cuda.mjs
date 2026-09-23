// Runs a command with the locally installed CUDA runtime (.cuda-libs, see docs) on LD_LIBRARY_PATH,
// so the engine can use the NVIDIA GPU. Without .cuda-libs the command runs unchanged (CPU).
//   node scripts/with-cuda.mjs <command> [args...]
import { spawn } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const base = path.join(root, '.cuda-libs', 'nvidia');
const libDirs = [];
const walk = (d, depth) => {
  if (depth > 3) return;
  for (const n of readdirSync(d)) {
    const p = path.join(d, n);
    if (!statSync(p).isDirectory()) continue;
    if (n === 'lib') libDirs.push(p);
    else walk(p, depth + 1);
  }
};
if (process.platform === 'linux' && existsSync(base)) walk(base, 0);

const env = { ...process.env };
if (libDirs.length) env.LD_LIBRARY_PATH = [...libDirs, env.LD_LIBRARY_PATH].filter(Boolean).join(':');
const [cmd, ...args] = process.argv.slice(2);
const child = spawn(cmd, args, { stdio: 'inherit', env });
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => child.kill(sig));
child.on('exit', (code, signal) => process.exit(signal ? 1 : (code ?? 0)));
