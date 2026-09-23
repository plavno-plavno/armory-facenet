// Loads/saves <dataDir>/config.json and applies validated partial patches.

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { EventEmitter } from 'node:events';
import { writeFileAtomic } from '../store/atomic.js';
import { EngineConfig, RESTART_REQUIRED } from './schema.js';

type Json = Record<string, unknown>;

function isObj(v: unknown): v is Json {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** Deep merge where arrays and scalars in `patch` replace values in `base`. */
export function deepMerge(base: Json, patch: Json): Json {
  const out: Json = { ...base };
  for (const [k, v] of Object.entries(patch)) {
    out[k] = isObj(v) && isObj(base[k]) ? deepMerge(base[k] as Json, v) : v;
  }
  return out;
}

function flatten(o: unknown, prefix = '', acc: Record<string, string> = {}): Record<string, string> {
  if (isObj(o)) for (const [k, v] of Object.entries(o)) flatten(v, prefix ? `${prefix}.${k}` : k, acc);
  else acc[prefix] = JSON.stringify(o);
  return acc;
}

export function changedPaths(a: unknown, b: unknown): string[] {
  const fa = flatten(a);
  const fb = flatten(b);
  const keys = new Set([...Object.keys(fa), ...Object.keys(fb)]);
  return [...keys].filter((k) => fa[k] !== fb[k]);
}

export function needsRestart(paths: string[]): string[] {
  return paths.filter((p) => RESTART_REQUIRED.some((r) => p === r || p.startsWith(`${r}.`)));
}

export class ConfigStore extends EventEmitter {
  private current: EngineConfig;
  /** Persisted settings; runtime overrides (CLI/env) are applied on top and never saved. */
  private disk: Json;
  /** Values the running engine was started with (restart-required keys). */
  private readonly startup: EngineConfig;
  private readonly file: string;

  constructor(
    dir: string,
    private readonly overrides: Json = {},
  ) {
    super();
    this.file = path.join(dir, 'config.json');
    this.disk = existsSync(this.file) ? (JSON.parse(readFileSync(this.file, 'utf8')) as Json) : {};
    this.current = EngineConfig.parse(deepMerge(this.disk, overrides));
    this.startup = this.current;
  }

  get(): EngineConfig {
    return this.current;
  }

  /** Keys changed since start that are only applied after a restart. */
  pendingRestart(): string[] {
    return needsRestart(changedPaths(this.startup, this.current));
  }

  async patch(patch: Json): Promise<{ config: EngineConfig; changed: string[]; restartRequired: string[] }> {
    const disk = deepMerge(this.disk, patch);
    EngineConfig.parse(disk); // validate what is persisted, even keys masked by runtime overrides (throws ZodError)
    const next = EngineConfig.parse(deepMerge(disk, this.overrides));
    const changed = changedPaths(this.current, next);
    this.disk = disk;
    this.current = next;
    await this.save();
    this.emit('change', next, changed);
    return { config: next, changed, restartRequired: this.pendingRestart() };
  }

  async save(): Promise<void> {
    await writeFileAtomic(this.file, Buffer.from(JSON.stringify(this.disk, null, 2)));
  }
}
