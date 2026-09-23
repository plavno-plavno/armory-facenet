import { v7 } from 'uuid';

export const newId = (): string => v7();
export const nowIso = (): string => new Date().toISOString();
