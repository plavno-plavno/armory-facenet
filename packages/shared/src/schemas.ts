// Shared zod schemas: single source of truth for the API and the admin UI (spec §5, §10).

import { z } from 'zod';

export const ErrorCode = z.enum([
  'VALIDATION_ERROR',
  'UNAUTHORIZED',
  'NOT_FOUND',
  'CONSENT_REQUIRED',
  'UNSUPPORTED_FORMAT',
  'FILE_TOO_LARGE',
  'IMAGE_TOO_SMALL',
  'NO_FACE',
  'MULTIPLE_FACES',
  'LOW_QUALITY',
  'DUPLICATE_SUSPECTED',
  'PHOTO_MISMATCH',
  'LAST_PHOTO',
  'VERSION_CONFLICT',
  'SOURCE_UNAVAILABLE',
  'MODEL_NOT_READY',
  'RATE_LIMITED',
  'INTERNAL',
]);
export type ErrorCode = z.infer<typeof ErrorCode>;

export const ApiErrorBody = z.object({
  error: z.object({ code: ErrorCode, message: z.string(), details: z.record(z.string(), z.unknown()) }),
});
export type ApiErrorBody = z.infer<typeof ApiErrorBody>;

export const BoxSchema = z.tuple([z.number(), z.number(), z.number(), z.number()]);

export const Consent = z.object({
  obtained: z.boolean(),
  obtainedAt: z.iso.datetime({ offset: true }).optional(),
  basis: z.string().max(500).optional(),
});

const CustomFieldValue = z.union([z.string().max(1000), z.number(), z.boolean()]);
export const CustomFields = z
  .record(z.string().min(1).max(100), CustomFieldValue)
  .refine((o) => Object.keys(o).length <= 50, { message: 'customFields: at most 50 keys' });

export const PersonStatus = z.enum(['active', 'disabled']);

export const PhotoQuality = z.object({
  detScore: z.number(),
  faceSize: z.number(),
  yaw: z.number(),
  rollDeg: z.number().optional(),
  sharpness: z.number(),
  brightness: z.number(),
});

export const Photo = z.object({
  id: z.string(),
  source: z.enum(['upload', 'camera']),
  createdAt: z.string(),
  faceBox: BoxSchema,
  quality: PhotoQuality,
  modelKeys: z.array(z.string()),
});
export type Photo = z.infer<typeof Photo>;

const personDataShape = {
  externalId: z.string().min(1).max(100).optional(),
  firstName: z.string().min(1).max(200),
  lastName: z.string().min(1).max(200),
  middleName: z.string().max(200).optional(),
  dateOfBirth: z.iso.date().optional(),
  department: z.string().max(200).optional(),
  position: z.string().max(200).optional(),
  notes: z.string().max(5000).optional(),
  customFields: CustomFields.optional(),
  status: PersonStatus.optional(),
  consent: Consent.optional(),
};

/** Body of `data` part in POST /persons. */
export const PersonCreate = z.object(personDataShape).strict();
export type PersonCreate = z.infer<typeof PersonCreate>;

/** PATCH /persons/:id. Nullable fields can be cleared with null. */
export const PersonPatch = z
  .object({
    externalId: personDataShape.externalId.nullable(),
    firstName: personDataShape.firstName,
    lastName: personDataShape.lastName,
    middleName: personDataShape.middleName.nullable(),
    dateOfBirth: personDataShape.dateOfBirth.nullable(),
    department: personDataShape.department.nullable(),
    position: personDataShape.position.nullable(),
    notes: personDataShape.notes.nullable(),
    customFields: CustomFields,
    status: PersonStatus,
    consent: Consent,
  })
  .partial()
  .strict();
export type PersonPatch = z.infer<typeof PersonPatch>;

export const Person = z.object({
  id: z.string(),
  ...personDataShape,
  status: PersonStatus,
  photos: z.array(Photo),
  createdAt: z.string(),
  updatedAt: z.string(),
  version: z.number().int(),
});
export type Person = z.infer<typeof Person>;

export const PersonList = z.object({
  items: z.array(Person),
  nextCursor: z.string().nullable(),
  total: z.number().int(),
});

export const RecognitionStatus = z.enum(['match', 'unknown', 'uncertain', 'low_quality', 'spoof']);
export type RecognitionStatus = z.infer<typeof RecognitionStatus>;

export const RecognitionEvent = z.object({
  type: z.literal('recognition.result'),
  eventId: z.string(),
  ts: z.string(),
  sourceId: z.string(),
  trackId: z.string(),
  status: RecognitionStatus,
  person: z
    .object({
      id: z.string(),
      externalId: z.string().optional(),
      firstName: z.string(),
      lastName: z.string(),
    })
    .optional(),
  personId: z.string().optional(),
  score: z.number().nullable(),
  secondScore: z.number().nullable(),
  frameAgreement: z.number().nullable(),
  framesUsed: z.number().int(),
  attempt: z.number().int(),
  latencyMs: z.number(),
  liveness: z.object({ live: z.boolean(), score: z.number() }).nullable(),
  snapshotId: z.string().optional(),
  reasons: z.record(z.string(), z.number()).optional(),
});
export type RecognitionEvent = z.infer<typeof RecognitionEvent>;

export const SourceState = z.enum(['starting', 'running', 'reconnecting', 'error', 'stopped']);
export type SourceState = z.infer<typeof SourceState>;

export const Roi = z.object({ x: z.number().min(0), y: z.number().min(0), w: z.number().positive(), h: z.number().positive() });

export const StreamInput = z.object({
  name: z.string().min(1).max(100),
  type: z.enum(['webcam', 'rtsp', 'file']),
  deviceId: z.string().optional(),
  url: z.string().optional(),
  enabled: z.boolean().default(true),
  detectFps: z.number().positive().max(60).optional(),
  roi: Roi.nullable().optional(),
  resolution: z.object({ width: z.number().int().positive(), height: z.number().int().positive() }).optional(),
  loop: z.boolean().optional(),
});
export type StreamInput = z.infer<typeof StreamInput>;

export const StreamInfo = StreamInput.extend({
  id: z.string(),
  status: SourceState,
  statusMessage: z.string().optional(),
  fps: z.number(),
  droppedFrames: z.number().int(),
  activeTracks: z.number().int(),
});
export type StreamInfo = z.infer<typeof StreamInfo>;

export const StreamStatusEvent = z.object({
  type: z.literal('stream.status'),
  ts: z.string(),
  sourceId: z.string(),
  status: SourceState,
  message: z.string().optional(),
});

export const IndexProgressEvent = z.object({
  type: z.literal('index.progress'),
  ts: z.string(),
  jobId: z.string(),
  done: z.number().int(),
  total: z.number().int(),
  state: z.enum(['running', 'completed', 'failed']),
});

export const PersonChangedEvent = z.object({
  type: z.literal('person.changed'),
  ts: z.string(),
  personId: z.string(),
  action: z.enum(['created', 'updated', 'deleted', 'photos_added', 'photo_deleted']),
});

export type EngineEvent =
  | RecognitionEvent
  | z.infer<typeof StreamStatusEvent>
  | z.infer<typeof IndexProgressEvent>
  | z.infer<typeof PersonChangedEvent>;

export const Job = z.object({
  id: z.string(),
  kind: z.string(),
  state: z.enum(['running', 'completed', 'failed']),
  done: z.number().int(),
  total: z.number().int(),
  errors: z.array(z.object({ personId: z.string(), photoId: z.string(), message: z.string() })),
  startedAt: z.string(),
  finishedAt: z.string().nullable(),
});
export type Job = z.infer<typeof Job>;
