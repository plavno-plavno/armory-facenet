import type { FastifyRequest } from 'fastify';
import { ApiError } from '../util/errors.js';

export interface ParsedMultipart {
  fields: Record<string, string>;
  files: Buffer[];
}

/** Read all parts; files from any of `fileFields` are collected in order. */
export async function readMultipart(req: FastifyRequest, fileFields: string[]): Promise<ParsedMultipart> {
  if (!req.isMultipart()) throw new ApiError('VALIDATION_ERROR', 'Expected multipart/form-data', {}, 415);
  const fields: Record<string, string> = {};
  const files: Buffer[] = [];
  for await (const part of req.parts()) {
    if (part.type === 'file') {
      const buf = await part.toBuffer(); // throws FST_REQ_FILE_TOO_LARGE on limit
      if (fileFields.includes(part.fieldname)) files.push(buf);
    } else {
      fields[part.fieldname] = String(part.value);
    }
  }
  return { fields, files };
}

export function parseJsonField<T>(raw: string | undefined, name: string): T | undefined {
  if (raw === undefined || raw === '') return undefined;
  try {
    return JSON.parse(raw) as T;
  } catch {
    throw new ApiError('VALIDATION_ERROR', `Field "${name}" must be valid JSON`);
  }
}

export function parseBool(v: unknown): boolean {
  return v === true || v === 'true' || v === '1';
}

/** faceBox as JSON array "[x,y,w,h]" or "x,y,w,h". */
export function parseFaceBox(v: string | undefined): [number, number, number, number] | undefined {
  if (!v) return undefined;
  const arr = v.trim().startsWith('[') ? parseJsonField<number[]>(v, 'faceBox') : v.split(',').map(Number);
  if (!arr || arr.length !== 4 || arr.some((n) => !Number.isFinite(n)) || arr[2] <= 0 || arr[3] <= 0) {
    throw new ApiError('VALIDATION_ERROR', 'faceBox must be [x, y, w, h]');
  }
  return arr as [number, number, number, number];
}

/** photoSources: JSON array of "upload" | "camera", one per photo. */
export function parsePhotoSources(v: string | undefined, count: number): ('upload' | 'camera')[] | undefined {
  const arr = parseJsonField<unknown[]>(v, 'photoSources');
  if (arr === undefined) return undefined;
  if (!Array.isArray(arr) || arr.length !== count || arr.some((x) => x !== 'upload' && x !== 'camera')) {
    throw new ApiError('VALIDATION_ERROR', 'photoSources must be an array of "upload" | "camera", one per photo');
  }
  return arr as ('upload' | 'camera')[];
}
