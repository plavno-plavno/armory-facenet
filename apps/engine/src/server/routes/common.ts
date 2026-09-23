import { ApiErrorBody } from '@faceid/shared';
import { z } from 'zod';

export const errors = (...codes: number[]) => Object.fromEntries(codes.map((c) => [c, ApiErrorBody]));
export const IdParams = z.object({ id: z.string().uuid() });
export const BoolFlag = z.enum(['true', 'false', '1', '0']).optional();
