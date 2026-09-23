import type { ErrorCode } from '@faceid/shared';

const STATUS: Record<ErrorCode, number> = {
  VALIDATION_ERROR: 400,
  UNAUTHORIZED: 401,
  NOT_FOUND: 404,
  CONSENT_REQUIRED: 400,
  UNSUPPORTED_FORMAT: 415,
  FILE_TOO_LARGE: 413,
  IMAGE_TOO_SMALL: 422,
  NO_FACE: 422,
  MULTIPLE_FACES: 422,
  LOW_QUALITY: 422,
  DUPLICATE_SUSPECTED: 409,
  PHOTO_MISMATCH: 422,
  LAST_PHOTO: 409,
  VERSION_CONFLICT: 412,
  SOURCE_UNAVAILABLE: 409,
  MODEL_NOT_READY: 503,
  RATE_LIMITED: 429,
  INTERNAL: 500,
};

export class ApiError extends Error {
  readonly status: number;
  constructor(
    readonly code: ErrorCode,
    message: string,
    readonly details: Record<string, unknown> = {},
    status?: number,
  ) {
    super(message);
    this.status = status ?? STATUS[code];
  }

  toBody() {
    return { error: { code: this.code, message: this.message, details: this.details } };
  }
}
