// Stale capture processes are released before the camera is opened; other programs are not touched.
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { deviceHolders, releaseDevice } from '../../src/pipeline/sources/device-lock.js';

const dir = mkdtempSync(path.join(tmpdir(), 'faceid-dev-'));
const device = path.join(dir, 'video9');
writeFileSync(device, '');
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const holdScript = `import time; f = open('${device}'); time.sleep(120)`; // tmp path: no quotes inside

/** A process whose argv looks like the engine's capture ffmpeg (argv[0] renamed with exec -a). */
function fakeCapture(name: string, args: string[]) {
  // bash stays as the parent, so the fake is not a child of this (test) process; bash is then suspended
  // like an engine stopped with Ctrl+Z.
  const quoted = args.map((a) => `'${a.replace(/'/g, `'\\''`)}'`).join(' ');
  const p = spawn('bash', ['-c', `(exec -a ${name} python3 -c "${holdScript}" ${quoted}) & wait`], { stdio: 'ignore', detached: true });
  return p;
}

const waitHolders = async (n: number) => {
  for (let i = 0; i < 50 && deviceHolders(device).length < n; i++) await new Promise((r) => setTimeout(r, 100));
};

describe.runIf(process.platform === 'linux')('device lock release', () => {
  it('stops a stale capture ffmpeg of a suspended engine but leaves other programs alone', async () => {
    const stale = fakeCapture('ffmpeg', ['-hide_banner', '-f', 'v4l2', '-input_format', 'mjpeg', '-i', device, '-an', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-']);
    const foreign = fakeCapture('zoom', ['--camera', device]);
    await waitHolders(2);
    process.kill(stale.pid!, 'SIGSTOP'); // the "engine" (parent) is suspended, like Ctrl+Z

    const before = deviceHolders(device);
    expect(before.find((h) => h.ours)).toBeTruthy();
    expect(before.find((h) => h.cmd.startsWith('zoom'))?.ours).toBe(false);

    const logs: string[] = [];
    const rest = await releaseDevice(device, (m) => logs.push(m));
    expect(rest.map((h) => h.cmd.split(' ')[0])).toEqual(['zoom']);
    expect(logs.join('\n')).toMatch(/stopping stale ffmpeg/);

    for (const p of [stale, foreign]) {
      try {
        process.kill(-p.pid!, 'SIGKILL');
      } catch {
        /* already gone: the released capture's parent exits on its own */
      }
    }
  });
});
