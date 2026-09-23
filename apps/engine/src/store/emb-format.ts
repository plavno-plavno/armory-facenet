// Embedding file format (spec §8.2), before encryption:
// "FEMB" | version u16 LE = 1 | dim u16 LE | reserved(8) | float32 LE[dim]

const MAGIC = 0x424d4546; // "FEMB" read as u32 LE
const HEADER = 16;

export function encodeEmbedding(v: Float32Array): Buffer {
  const buf = Buffer.alloc(HEADER + v.length * 4);
  buf.writeUInt32LE(MAGIC, 0);
  buf.writeUInt16LE(1, 4);
  buf.writeUInt16LE(v.length, 6);
  for (let i = 0; i < v.length; i++) buf.writeFloatLE(v[i], HEADER + i * 4);
  return buf;
}

export function decodeEmbedding(buf: Buffer): Float32Array {
  if (buf.length < HEADER || buf.readUInt32LE(0) !== MAGIC) throw new Error('Not an embedding file');
  const version = buf.readUInt16LE(4);
  if (version !== 1) throw new Error(`Unsupported embedding version ${version}`);
  const dim = buf.readUInt16LE(6);
  if (buf.length !== HEADER + dim * 4) throw new Error('Embedding file size mismatch');
  const out = new Float32Array(dim);
  for (let i = 0; i < dim; i++) out[i] = buf.readFloatLE(HEADER + i * 4);
  return out;
}
