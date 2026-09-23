// NVIDIA CUDA runtime discovery for the ONNX Runtime CUDA provider (Linux).
//
// onnxruntime-node's CUDA provider links against CUDA 13 (cudart, cuBLAS) and dlopens cuDNN 9.
// LD_LIBRARY_PATH cannot be changed after the process has started, so libraries shipped next to
// the app (.cuda-libs from `npm run cuda:install`, FACEID_CUDA_LIBS, or <resources>/cuda-libs) are
// preloaded with RTLD_GLOBAL: the loader then satisfies the provider's dependencies by soname.

import { execFileSync } from 'node:child_process';
import { existsSync, readdirSync, statSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const RTLD_NODELETE = 0x1000; // glibc: keep the library mapped even if Node closes the handle
const REQUIRED = ['libcudart.so.13', 'libcublasLt.so.13', 'libcublas.so.13', 'libcudnn.so.9'];

export interface CudaStatus {
  /** An NVIDIA driver is present (so trying the CUDA provider makes sense). */
  gpu: boolean;
  /** All required runtime libraries are resolvable (system or preloaded). */
  runtime: boolean;
  /** Directories preloaded from. */
  preloadedFrom: string[];
  missing: string[];
}

let status: CudaStatus | null = null;

function candidateRoots(): string[] {
  const roots: string[] = [];
  if (process.env.FACEID_CUDA_LIBS) roots.push(...process.env.FACEID_CUDA_LIBS.split(path.delimiter));
  const res = (process as { resourcesPath?: string }).resourcesPath;
  if (res) roots.push(path.join(res, 'cuda-libs'));
  // .cuda-libs in the working directory or any parent of this module (dev checkout).
  roots.push(path.join(process.cwd(), '.cuda-libs'));
  let dir = path.dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 6; i++) {
    roots.push(path.join(dir, '.cuda-libs'));
    dir = path.dirname(dir);
  }
  return [...new Set(roots)].filter((r) => existsSync(r));
}

/** All directories named "lib" (pip nvidia-* wheels) or containing .so files under a root. */
function libDirs(root: string): string[] {
  const out: string[] = [];
  const walk = (d: string, depth: number) => {
    let names: string[];
    try {
      names = readdirSync(d);
    } catch {
      return;
    }
    if (names.some((n) => /\.so(\.\d+)*$/.test(n))) out.push(d);
    if (depth >= 4) return;
    for (const n of names) {
      const p = path.join(d, n);
      try {
        if (statSync(p).isDirectory()) walk(p, depth + 1);
      } catch {
        /* ignore */
      }
    }
  };
  walk(root, 0);
  return out;
}

function systemLibs(): string {
  for (const bin of ['ldconfig', '/sbin/ldconfig', '/usr/sbin/ldconfig']) {
    try {
      return execFileSync(bin, ['-p'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
    } catch {
      /* try next */
    }
  }
  return '';
}

/** dlopen a shared library globally; returns false if it could not be mapped. */
function preload(file: string): boolean {
  const flags = os.constants.dlopen.RTLD_NOW | os.constants.dlopen.RTLD_GLOBAL | RTLD_NODELETE;
  try {
    process.dlopen({ exports: {} } as NodeJS.Module, file, flags);
    return true;
  } catch (e) {
    // A plain shared library is not a Node addon: it loads, then registration fails. That's fine.
    return /self-register|napi_register|Module did not/i.test((e as Error).message);
  }
}

/**
 * Detect the GPU and make the CUDA runtime available to ONNX Runtime. Idempotent.
 */
export function prepareCuda(): CudaStatus {
  if (status) return status;
  const gpu = process.platform === 'linux' ? existsSync('/proc/driver/nvidia/version') : process.platform === 'win32';
  const sys = process.platform === 'linux' ? systemLibs() : '';
  const envDirs = (process.env.LD_LIBRARY_PATH ?? '').split(':').filter(Boolean);
  const found = (lib: string) => sys.includes(lib) || envDirs.some((d) => existsSync(path.join(d, lib)));

  const preloadedFrom: string[] = [];
  if (process.platform === 'linux' && gpu && !REQUIRED.every(found)) {
    const dirs = candidateRoots().flatMap(libDirs);
    // Only real sonames (libX.so.N), not static archives or linker names.
    const files = dirs.flatMap((d) => readdirSync(d).filter((n) => /^lib(cudart|cublasLt|cublas|cudnn[a-z_]*|nvrtc)\.so\.\d+$/.test(n)).map((n) => path.join(d, n)));
    // Dependencies first (cudart before cuBLAS, cudnn_graph before the other cuDNN parts); multiple passes
    // resolve any remaining ordering.
    const rank = (f: string) => ['cudart', 'nvrtc', 'cublasLt', 'cublas', 'cudnn_graph', 'cudnn.so', 'cudnn'].findIndex((k) => path.basename(f).includes(k));
    let pending = files.sort((a, b) => rank(a) - rank(b));
    const loaded = new Set<string>();
    for (let pass = 0; pass < 4 && pending.length; pass++) {
      pending = pending.filter((f) => {
        if (!preload(f)) return true;
        loaded.add(path.basename(f));
        if (!preloadedFrom.includes(path.dirname(f))) preloadedFrom.push(path.dirname(f));
        return false;
      });
    }
    const missing = REQUIRED.filter((l) => !found(l) && !loaded.has(l));
    status = { gpu, runtime: missing.length === 0, preloadedFrom, missing };
    return status;
  }
  const missing = REQUIRED.filter((l) => !found(l));
  status = { gpu, runtime: process.platform !== 'linux' || missing.length === 0, preloadedFrom, missing };
  return status;
}
