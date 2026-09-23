// Minimal typed client for the engine REST API (used by the admin UI and tests).

import type { Job, Person, PersonCreate, PersonPatch, StreamInfo, StreamInput } from './schemas.js';

export class ApiClientError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
    public readonly details: Record<string, unknown> = {},
  ) {
    super(message);
  }
}

export interface ClientOptions {
  baseUrl: string; // e.g. http://127.0.0.1:47810/api/v1
  token: string;
  fetch?: typeof fetch;
}

export class FaceIdClient {
  private readonly f: typeof fetch;
  constructor(private readonly opts: ClientOptions) {
    // Bound: calling an unbound window.fetch reference throws "Illegal invocation" in browsers.
    this.f = opts.fetch ?? fetch.bind(globalThis);
  }

  async request<T>(method: string, path: string, body?: unknown, headers: Record<string, string> = {}): Promise<T> {
    const init: RequestInit = { method, headers: { Authorization: `Bearer ${this.opts.token}`, ...headers } };
    if (body instanceof FormData) init.body = body;
    else if (body !== undefined) {
      init.body = JSON.stringify(body);
      (init.headers as Record<string, string>)['Content-Type'] = 'application/json';
    }
    const res = await this.f(`${this.opts.baseUrl}${path}`, init);
    if (res.status === 204) return undefined as T;
    const ct = res.headers.get('content-type') ?? '';
    if (!res.ok) {
      const j = ct.includes('json') ? await res.json() : { error: { code: 'INTERNAL', message: await res.text(), details: {} } };
      throw new ApiClientError(res.status, j.error?.code, j.error?.message, j.error?.details);
    }
    return (ct.includes('json') ? res.json() : res.blob()) as Promise<T>;
  }

  health() {
    return this.request<Record<string, any>>('GET', '/health');
  }
  listPersons(params: { q?: string; status?: string; limit?: number; cursor?: string } = {}) {
    const qs = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined && v !== '') as [string, string][]);
    return this.request<{ items: Person[]; nextCursor: string | null; total: number }>('GET', `/persons?${qs}`);
  }
  getPerson(id: string) {
    return this.request<Person>('GET', `/persons/${id}`);
  }
  createPerson(data: PersonCreate, photos: Blob[], opts: { allowDuplicate?: boolean } = {}) {
    const fd = new FormData();
    fd.append('data', JSON.stringify(data));
    photos.forEach((p, i) => fd.append('photos', p, `photo${i}.jpg`));
    return this.request<Person>('POST', `/persons${opts.allowDuplicate ? '?allowDuplicate=true' : ''}`, fd);
  }
  patchPerson(id: string, patch: PersonPatch, version?: number) {
    return this.request<Person>('PATCH', `/persons/${id}`, patch, version !== undefined ? { 'If-Match': String(version) } : {});
  }
  deletePerson(id: string, purgeEvents = false) {
    return this.request<void>('DELETE', `/persons/${id}${purgeEvents ? '?purgeEvents=true' : ''}`);
  }
  addPhotos(id: string, photos: Blob[], opts: { force?: boolean } = {}) {
    const fd = new FormData();
    photos.forEach((p, i) => fd.append('photos', p, `photo${i}.jpg`));
    return this.request<unknown[]>('POST', `/persons/${id}/photos${opts.force ? '?force=true' : ''}`, fd);
  }
  capturePhotos(id: string, sourceId: string) {
    return this.request<unknown[]>('POST', `/persons/${id}/photos/capture`, { sourceId });
  }
  /** Best enrollment-quality frames from a running stream (nothing is stored). */
  captureFromStream(streamId: string) {
    return this.request<{ frames: { jpeg: string; width: number; height: number; faceBox: number[]; quality: Record<string, number> }[] }>(
      'POST',
      `/streams/${streamId}/capture`,
    );
  }
  /** On-demand recognition of the person in front of a running stream's camera. */
  recognizeOnStream(streamId: string) {
    return this.request<{
      event: import('./schemas.js').RecognitionEvent;
      face: string; // aligned 112x112 PNG, base64
      faceBox: number[];
      candidates: { personId: string; name: string; score: number; photoId: string }[];
      framesSeen: number;
      goodFrames: number;
    }>('POST', `/streams/${streamId}/recognize`);
  }
  deletePhoto(id: string, photoId: string) {
    return this.request<void>('DELETE', `/persons/${id}/photos/${photoId}`);
  }
  listStreams() {
    return this.request<StreamInfo[]>('GET', '/streams');
  }
  createStream(input: StreamInput) {
    return this.request<StreamInfo>('POST', '/streams', input);
  }
  deleteStream(id: string) {
    return this.request<void>('DELETE', `/streams/${id}`);
  }
  streamAction(id: string, action: 'start' | 'stop') {
    return this.request<StreamInfo>('POST', `/streams/${id}/${action}`);
  }
  cameras() {
    return this.request<{ deviceId: string; label: string }[]>('GET', '/devices/cameras');
  }
  getConfig() {
    return this.request<{ config: Record<string, any>; restartRequired: string[] }>('GET', '/config');
  }
  patchConfig(patch: Record<string, unknown>) {
    return this.request<{ config: Record<string, any>; restartRequired: string[] }>('PATCH', '/config', patch);
  }
  recognitions(params: Record<string, string> = {}) {
    return this.request<{ items: any[]; nextCursor: string | null }>('GET', `/events/recognitions?${new URLSearchParams(params)}`);
  }
  job(id: string) {
    return this.request<Job>('GET', `/admin/jobs/${id}`);
  }
}
