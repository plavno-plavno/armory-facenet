// Image decode/encode via sharp.

import sharp, { type Metadata } from 'sharp';
import { swapRB } from './preprocess.js';
import type { Frame } from './types.js';

export interface DecodedImage {
  frame: Frame; // BGR24
  format: string;
  /** Oriented image re-encoded as JPEG without metadata (for storage). */
  jpeg: Buffer;
}

export const SUPPORTED_FORMATS = new Set(['jpeg', 'png', 'webp']);

export class UnsupportedImageError extends Error {}

/** Decode an uploaded image, applying EXIF orientation and stripping metadata. */
export async function decodeImage(buf: Buffer, sourceId = 'upload'): Promise<DecodedImage> {
  let meta: Metadata;
  try {
    meta = await sharp(buf).metadata();
  } catch {
    throw new UnsupportedImageError('Cannot decode image');
  }
  if (!meta.format || !SUPPORTED_FORMATS.has(meta.format)) {
    throw new UnsupportedImageError(`Unsupported format: ${meta.format ?? 'unknown'}`);
  }
  const oriented = sharp(buf).rotate();
  const { data, info } = await oriented.clone().removeAlpha().toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
  if (info.channels !== 3) throw new UnsupportedImageError(`Unexpected channel count ${info.channels}`);
  const jpeg = await oriented.clone().jpeg({ quality: 95 }).toBuffer(); // sharp drops metadata by default
  return {
    frame: { data: swapRB(new Uint8Array(data.buffer, data.byteOffset, data.length)), width: info.width, height: info.height, ts: 0, sourceId },
    format: meta.format,
    jpeg,
  };
}

export async function encodeJpegBgr(bgr: Uint8Array, width: number, height: number, quality = 90, maxWidth?: number): Promise<Buffer> {
  let img = sharp(Buffer.from(swapRB(bgr)), { raw: { width, height, channels: 3 } });
  if (maxWidth && maxWidth < width) img = img.resize({ width: maxWidth });
  return img.jpeg({ quality }).toBuffer();
}

export async function encodePngRgb(rgb: Uint8Array, width: number, height: number): Promise<Buffer> {
  return sharp(Buffer.from(rgb), { raw: { width, height, channels: 3 } }).png().toBuffer();
}
