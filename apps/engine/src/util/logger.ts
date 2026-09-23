// Application logger (pino). Must never receive names, personal data, embeddings or images
// (spec §12.4); known PII keys are redacted as a second line of defence.

import { mkdirSync } from 'node:fs';
import path from 'node:path';
import pino from 'pino';

export type Logger = pino.Logger;

export const PII_KEYS = [
  'firstName',
  'lastName',
  'middleName',
  'dateOfBirth',
  'notes',
  'customFields',
  'department',
  'position',
  'externalId',
  'person',
  'embedding',
  'authorization',
  'token',
];

export function createLogger(opts: { level: string; dataDir?: string; pretty?: boolean }): Logger {
  const redact = { paths: [...PII_KEYS, ...PII_KEYS.map((k) => `*.${k}`), 'req.headers.authorization'], censor: '[redacted]' };
  if (!opts.dataDir) return pino({ level: opts.level, redact });
  const dir = path.join(opts.dataDir, 'logs', 'app');
  mkdirSync(dir, { recursive: true });
  const transport = pino.transport({
    targets: [
      { target: 'pino-roll', level: opts.level, options: { file: path.join(dir, 'engine'), frequency: 'daily', size: '20m', mkdir: true, limit: { count: 14 }, extension: '.log' } },
      { target: 'pino/file', level: opts.level, options: { destination: 1 } },
    ],
  });
  return pino({ level: opts.level, redact }, transport);
}
