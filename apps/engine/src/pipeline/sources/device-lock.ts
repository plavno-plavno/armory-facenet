// Linux: free a V4L2 device held by a stale ffmpeg of a previous engine run (e.g. an engine that was
// suspended with Ctrl+Z or killed while its ffmpeg child kept running). Other applications are never
// touched; they are only reported so the stream status can say who holds the camera.

import { readdirSync, readFileSync, readlinkSync, realpathSync } from 'node:fs';

export interface DeviceHolder {
  pid: number;
  ppid: number;
  cmd: string;
  /** An ffmpeg started by a FaceID engine (our rawvideo/bgr24 capture signature). */
  ours: boolean;
}

function readProc(pid: number): { cmdline: string[]; ppid: number } | null {
  try {
    const cmdline = readFileSync(`/proc/${pid}/cmdline`, 'utf8').split('\0').filter(Boolean);
    const ppid = Number(/PPid:\s+(\d+)/.exec(readFileSync(`/proc/${pid}/status`, 'utf8'))?.[1] ?? 0);
    return { cmdline, ppid };
  } catch {
    return null;
  }
}

/** Processes with an open file descriptor on `device`. */
export function deviceHolders(device: string): DeviceHolder[] {
  if (process.platform !== 'linux') return [];
  let target: string;
  try {
    target = realpathSync(device);
  } catch {
    return [];
  }
  const out: DeviceHolder[] = [];
  let pids: string[] = [];
  try {
    pids = readdirSync('/proc').filter((n) => /^\d+$/.test(n));
  } catch {
    return [];
  }
  for (const p of pids) {
    const pid = Number(p);
    if (pid === process.pid) continue;
    let fds: string[];
    try {
      fds = readdirSync(`/proc/${pid}/fd`); // only our own user's processes are readable
    } catch {
      continue;
    }
    const holds = fds.some((fd) => {
      try {
        return readlinkSync(`/proc/${pid}/fd/${fd}`) === target;
      } catch {
        return false;
      }
    });
    if (!holds) continue;
    const info = readProc(pid);
    if (!info) continue;
    const a = info.cmdline;
    const exe = a[0]?.split('/').pop() ?? '';
    const ours =
      /^ffmpeg(\.exe)?$/.test(exe) &&
      a.includes('v4l2') &&
      a[a.indexOf('-i') + 1] !== undefined &&
      realpathOr(a[a.indexOf('-i') + 1]) === target &&
      a.includes('rawvideo') &&
      a.includes('bgr24') &&
      a[a.length - 1] === '-';
    out.push({ pid, ppid: info.ppid, cmd: a.join(' ').slice(0, 200), ours });
  }
  return out;
}

function realpathOr(p: string): string {
  try {
    return realpathSync(p);
  } catch {
    return p;
  }
}

const alive = (pid: number) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

/**
 * Terminate stale capture ffmpegs of other engine instances holding `device`. ffmpegs that belong to
 * this engine (a second stream on the same camera) are left alone. Returns the remaining holders.
 */
export async function releaseDevice(device: string, log?: (msg: string) => void): Promise<DeviceHolder[]> {
  const holders = deviceHolders(device);
  const stale = holders.filter((h) => h.ours && h.ppid !== process.pid);
  for (const h of stale) {
    log?.(`releasing ${device}: stopping stale ffmpeg pid ${h.pid} (parent ${h.ppid})`);
    try {
      process.kill(h.pid, 'SIGCONT'); // a suspended process does not act on SIGTERM until resumed
      process.kill(h.pid, 'SIGTERM');
    } catch {
      /* already gone */
    }
  }
  if (stale.length) {
    for (let i = 0; i < 20 && stale.some((h) => alive(h.pid)); i++) await new Promise((r) => setTimeout(r, 100));
    for (const h of stale) {
      if (alive(h.pid)) {
        try {
          process.kill(h.pid, 'SIGKILL');
        } catch {
          /* gone */
        }
      }
    }
    await new Promise((r) => setTimeout(r, 200)); // let the kernel release the device
  }
  return deviceHolders(device);
}

/** All V4L2 capture nodes. */
export function videoDevices(): string[] {
  if (process.platform !== 'linux') return [];
  try {
    return readdirSync('/dev')
      .filter((n) => /^video\d+$/.test(n))
      .map((n) => `/dev/${n}`);
  } catch {
    return [];
  }
}

/** Startup: free every camera held by a stale ffmpeg of an earlier engine run. */
export async function releaseStaleCaptures(log?: (msg: string) => void): Promise<void> {
  for (const d of videoDevices()) await releaseDevice(d, log);
}

/** Human-readable description of foreign holders for the stream status. */
export function describeHolders(holders: DeviceHolder[]): string {
  return holders.map((h) => `${h.cmd.split(' ')[0].split('/').pop()} (pid ${h.pid})`).join(', ');
}

// ---------- API port held by a suspended engine ----------

function processState(pid: number): string {
  try {
    return /State:\s+(\S)/.exec(readFileSync(`/proc/${pid}/status`, 'utf8'))?.[1] ?? '?';
  } catch {
    return '?';
  }
}

/** PIDs (of our user) listening on a local TCP port. */
function portHolders(port: number): number[] {
  const inodes = new Set<string>();
  for (const f of ['/proc/net/tcp', '/proc/net/tcp6']) {
    try {
      for (const line of readFileSync(f, 'utf8').split('\n').slice(1)) {
        const c = line.trim().split(/\s+/);
        if (c.length > 9 && c[3] === '0A' && parseInt(c[1].split(':')[1], 16) === port) inodes.add(c[9]);
      }
    } catch {
      /* no procfs */
    }
  }
  if (!inodes.size) return [];
  const pids: number[] = [];
  for (const p of readdirSync('/proc').filter((n) => /^\d+$/.test(n))) {
    try {
      for (const fd of readdirSync(`/proc/${p}/fd`)) {
        const m = /^socket:\[(\d+)\]$/.exec(readlinkSync(`/proc/${p}/fd/${fd}`));
        if (m && inodes.has(m[1])) {
          pids.push(Number(p));
          break;
        }
      }
    } catch {
      /* not ours / gone */
    }
  }
  return pids;
}

/**
 * If `port` is held by a *suspended* (Ctrl+Z) FaceID engine, resume it with SIGINT so it shuts down
 * gracefully (and releases its cameras). A running engine or any other program is left alone.
 * Returns a description of what still holds the port, or null if it is free.
 */
export async function releaseStalePort(port: number, log?: (msg: string) => void): Promise<string | null> {
  if (process.platform !== 'linux') return null;
  const isEngine = (pid: number) => {
    const cmd = readProc(pid)?.cmdline.join(' ') ?? '';
    return /engine[\/](src[\/]main\.ts|dist[\/]main\.js)|engine\.mjs/.test(cmd);
  };
  for (const pid of portHolders(port)) {
    if (pid !== process.pid && isEngine(pid) && processState(pid) === 'T') {
      log?.(`port ${port} is held by a suspended engine (pid ${pid}); stopping it`);
      try {
        process.kill(pid, 'SIGCONT');
        process.kill(pid, 'SIGINT');
      } catch {
        /* gone */
      }
      for (let i = 0; i < 50 && alive(pid); i++) await new Promise((r) => setTimeout(r, 100));
      if (alive(pid)) {
        try {
          process.kill(pid, 'SIGKILL');
        } catch {
          /* gone */
        }
      }
    }
  }
  await new Promise((r) => setTimeout(r, 200));
  const rest = portHolders(port).filter((p) => p !== process.pid);
  if (!rest.length) return null;
  return rest.map((pid) => `pid ${pid}${isEngine(pid) ? ' (another FaceID engine is running)' : ''}: ${(readProc(pid)?.cmdline.join(' ') ?? '').slice(0, 120)}`).join('; ');
}
