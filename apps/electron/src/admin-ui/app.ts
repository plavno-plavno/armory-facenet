// Admin UI (spec §2, §14): talks to the engine only through the public REST API / WebSocket.

import { ApiClientError, FaceIdClient, type Person, type StreamInfo } from '@faceid/shared';

interface Conn {
  baseUrl: string;
  token: string;
}
declare global {
  interface Window {
    faceid: {
      getConn(): Promise<Conn | null>;
      getStatus(): Promise<{ status: string; error: string }>;
      onConn(cb: (c: Conn) => void): void;
      onStatus(cb: (s: { status: string; error: string }) => void): void;
      submitPassword(pw: string): void;
    };
  }
}

/** Opened in a plain browser (no Electron preload bridge). */
const webMode = !('faceid' in window);
let conn: Conn | null = null;
let api: FaceIdClient;
let ws: WebSocket | null = null;
const liveEvents: any[] = [];
const eventListeners = new Set<(e: any) => void>();
let cleanup: (() => void)[] = [];

// ---------- tiny DOM helpers ----------

type Child = Node | string | number | null | undefined | false;
function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Record<string, any> = {}, ...children: (Child | Child[])[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === 'class') el.className = v;
    else if (k in el && k !== 'list') (el as any)[k] = v;
    else el.setAttribute(k, String(v));
  }
  for (const c of children.flat()) if (c !== null && c !== undefined && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}
const view = () => document.getElementById('view')!;
function render(...nodes: Child[]) {
  view().replaceChildren(...(nodes.filter(Boolean) as Node[]));
}
function toast(msg: string, ms = 3500) {
  const t = document.getElementById('toast')!;
  t.textContent = msg;
  t.classList.remove('hidden');
  setTimeout(() => t.classList.add('hidden'), ms);
}
const badge = (s: string, text?: string) => h('span', { class: `badge ${s}` }, text ?? s);
const fmtTime = (iso: string) => new Date(iso).toLocaleString('en-US');
const personName = (p: Pick<Person, 'lastName' | 'firstName' | 'middleName'>) => [p.lastName, p.firstName, p.middleName].filter(Boolean).join(' ');

function errText(e: unknown): string {
  if (e instanceof ApiClientError) {
    const reasons = (e.details?.reasons as string[] | undefined)?.join(', ');
    return `${e.code}: ${e.message}${reasons ? ` (${reasons})` : ''}`;
  }
  return (e as Error).message;
}

async function authedBlobUrl(path: string): Promise<string> {
  const r = await fetch(`${conn!.baseUrl}${path}`, { headers: { Authorization: `Bearer ${conn!.token}` } });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return URL.createObjectURL(await r.blob());
}

function authedImg(path: string, cls = ''): HTMLImageElement {
  const img = h('img', { class: cls, alt: '' });
  authedBlobUrl(path)
    .then((u) => {
      img.src = u;
      img.onload = () => URL.revokeObjectURL(u);
    })
    .catch(() => undefined);
  return img;
}

// ---------- connection & events ----------

function connectWs() {
  ws?.close();
  if (!conn) return;
  ws = new WebSocket(conn.baseUrl.replace(/^http/, 'ws') + '/events', [`bearer.${conn.token}`]);
  ws.onmessage = (m) => {
    const e = JSON.parse(m.data);
    if (e.type === 'recognition.result') {
      liveEvents.unshift(e);
      liveEvents.length = Math.min(liveEvents.length, 200);
    }
    eventListeners.forEach((fn) => fn(e));
  };
  ws.onclose = () => setTimeout(() => conn && ws?.readyState !== WebSocket.OPEN && connectWs(), 2000);
}

function setConn(c: Conn | null) {
  conn = c;
  const st = document.getElementById('engine-state')!;
  if (!c) {
    st.textContent = webMode ? 'not signed in' : 'engine starting…';
    return;
  }
  api = new FaceIdClient({ baseUrl: c.baseUrl, token: c.token });
  st.textContent = `API: ${c.baseUrl.replace('/api/v1', '')}`;
  connectWs();
  route();
}

// ---------- pages ----------

async function dashboard() {
  const hl = await api.health();
  const streams = await api.listStreams();
  const recent = h('tbody');
  const fillRecent = () =>
    recent.replaceChildren(
      ...liveEvents.slice(0, 12).map((e) =>
        h('tr', {}, h('td', {}, fmtTime(e.ts)), h('td', {}, badge(e.status)), h('td', {}, e.person ? personName(e.person) : '—'), h('td', {}, e.score ?? '—'), h('td', {}, `${e.latencyMs} ms`)),
      ),
    );
  fillRecent();
  const onEv = (e: any) => e.type === 'recognition.result' && fillRecent();
  eventListeners.add(onEv);
  cleanup.push(() => eventListeners.delete(onEv));

  const tiles = streams.filter((s) => s.status === 'running').map((s) => {
    const img = livePreview(() => s.id, 480);
    return h('div', { class: 'card' }, h('div', { class: 'row' }, h('b', {}, s.name), h('span', { class: 'spacer' }), badge(s.status)), img);
  });

  render(
    h('h1', {}, 'Status'),
    h(
      'div',
      { class: 'grid' },
      h('div', { class: 'card' }, h('div', { class: 'muted' }, 'State'), h('div', { class: 'stat' }, badge(hl.status, hl.status))),
      h('div', { class: 'card' }, h('div', { class: 'muted' }, 'Persons / embeddings'), h('div', { class: 'stat' }, `${hl.gallery.persons} / ${hl.gallery.embeddings}`)),
      h('div', { class: 'card' }, h('div', { class: 'muted' }, 'Latency p50 / p95'), h('div', { class: 'stat' }, `${hl.metrics.latencyP50 ?? '—'} / ${hl.metrics.latencyP95 ?? '—'} ms`)),
      h('div', { class: 'card' }, h('div', { class: 'muted' }, 'Detection FPS / dropped'), h('div', { class: 'stat' }, `${hl.metrics.detectFps} / ${hl.metrics.droppedFrames}`)),
    ),
    h(
      'div',
      { class: 'card' },
      h(
        'dl',
        { class: 'kv' },
        h('dt', {}, 'Detector'),
        h('dd', {}, `${hl.models.detector.id} (${hl.models.detector.executionProvider})`),
        h('dt', {}, 'Embedding model'),
        h('dd', {}, `${hl.models.embedder.modelKey} (${hl.models.embedder.executionProvider})`),
        h('dt', {}, 'Anti-spoofing'),
        h('dd', {}, hl.models.liveness?.enabled ? hl.models.liveness.models.join(' + ') : h('span', { class: 'err' }, 'off')),
        h('dt', {}, 'Uptime'),
        h('dd', {}, `${Math.round(hl.uptimeSec / 60)} min`),
        ...(hl.warnings as string[]).flatMap((w) => [h('dt', { class: 'err' }, 'Warning'), h('dd', { class: 'err' }, w)]),
        ...(hl.restartRequired.length ? [h('dt', {}, 'Restart required'), h('dd', {}, hl.restartRequired.join(', '))] : []),
      ),
    ),
    tiles.length ? h('div', { class: 'grid', style: 'grid-template-columns: repeat(auto-fill, minmax(420px, 1fr))' }, tiles) : null,
    h('h2', {}, 'Recent recognitions'),
    h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Time'), h('th', {}, 'Status'), h('th', {}, 'Person'), h('th', {}, 'Score'), h('th', {}, 'Latency'))), recent),
  );
}

async function persons(params: URLSearchParams) {
  const q = params.get('q') ?? '';
  const status = params.get('status') ?? '';
  const cursor = params.get('cursor') ?? undefined;
  const list = await api.listPersons({ q, status, limit: 50, cursor });
  const search = h('input', { placeholder: 'Search by name or externalId', value: q, style: 'width:320px' });
  const sel = h('select', {}, h('option', { value: '' }, 'all'), h('option', { value: 'active', selected: status === 'active' }, 'active'), h('option', { value: 'disabled', selected: status === 'disabled' }, 'disabled'));
  const go = (extra: Record<string, string> = {}) => (location.hash = `#/persons?${new URLSearchParams({ q: search.value, status: sel.value, ...extra })}`);
  search.addEventListener('keydown', (e) => e.key === 'Enter' && go());
  sel.addEventListener('change', () => go());
  render(
    h('div', { class: 'row' }, h('h1', {}, `Persons (${list.total})`), h('span', { class: 'spacer' }), h('button', { class: 'primary', onclick: () => (location.hash = '#/enroll') }, '+ Enroll')),
    h('div', { class: 'row', style: 'margin-bottom:12px' }, search, sel, h('button', { onclick: () => go() }, 'Search')),
    h(
      'table',
      {},
      h('thead', {}, h('tr', {}, h('th', {}, ''), h('th', {}, 'Name'), h('th', {}, 'externalId'), h('th', {}, 'Department'), h('th', {}, 'Photos'), h('th', {}, 'Status'))),
      h(
        'tbody',
        {},
        list.items.map((p) =>
          h(
            'tr',
            { class: 'click', onclick: () => (location.hash = `#/person/${p.id}`) },
            h('td', {}, p.photos[0] ? authedImg(`/persons/${p.id}/photos/${p.photos[0].id}?variant=aligned`, 'thumb') : ''),
            h('td', {}, personName(p)),
            h('td', {}, p.externalId ?? ''),
            h('td', {}, p.department ?? ''),
            h('td', {}, p.photos.length),
            h('td', {}, badge(p.status)),
          ),
        ),
      ),
    ),
    list.nextCursor ? h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { onclick: () => go({ cursor: list.nextCursor! }) }, 'Next →')) : null,
  );
}

function personForm(p?: Partial<Person>) {
  const f = {
    lastName: h('input', { value: p?.lastName ?? '', required: true }),
    firstName: h('input', { value: p?.firstName ?? '', required: true }),
    middleName: h('input', { value: p?.middleName ?? '' }),
    externalId: h('input', { value: p?.externalId ?? '' }),
    dateOfBirth: h('input', { type: 'date', value: p?.dateOfBirth ?? '' }),
    department: h('input', { value: p?.department ?? '' }),
    position: h('input', { value: p?.position ?? '' }),
    notes: h('textarea', { value: p?.notes ?? '' }),
  };
  const labels: Record<keyof typeof f, string> = {
    lastName: 'Last name *',
    firstName: 'First name *',
    middleName: 'Middle name',
    externalId: 'External ID',
    dateOfBirth: 'Date of birth',
    department: 'Department',
    position: 'Position',
    notes: 'Notes',
  };
  const node = h('div', { class: 'form' }, ...(Object.keys(f) as (keyof typeof f)[]).map((k) => h('label', { class: 'f' }, h('span', {}, labels[k]), f[k])));
  const values = (forPatch: boolean) => {
    const out: Record<string, unknown> = {};
    for (const [k, el] of Object.entries(f)) {
      const v = (el as HTMLInputElement).value.trim();
      if (v) out[k] = v;
      else if (forPatch && (p as any)?.[k] !== undefined) out[k] = null;
    }
    return out;
  };
  return { node, values };
}

async function personPage(id: string) {
  const p = await api.getPerson(id);
  const form = personForm(p);
  const fileInput = h('input', { type: 'file', accept: 'image/jpeg,image/png,image/webp', multiple: true });
  const streams = (await api.listStreams()).filter((s) => s.status === 'running');
  const streamSel = h('select', {}, streams.map((s) => h('option', { value: s.id }, s.name)));
  const msg = h('div', { class: 'err' });
  const purge = h('input', { type: 'checkbox' });

  const save = async () => {
    try {
      await api.patchPerson(id, form.values(true) as never, p.version);
      toast('Saved');
      route();
    } catch (e) {
      msg.textContent = errText(e);
    }
  };
  const addPhotos = async (force = false) => {
    if (!fileInput.files?.length) return;
    try {
      await api.addPhotos(id, [...fileInput.files], { force });
      toast('Photos added');
      route();
    } catch (e) {
      msg.replaceChildren(errText(e), e instanceof ApiClientError && e.code === 'PHOTO_MISMATCH' ? h('button', { onclick: () => addPhotos(true), style: 'margin-left:8px' }, 'Add anyway') : '');
    }
  };
  const capture = async () => {
    msg.textContent = 'Stand in front of the camera…';
    try {
      await api.capturePhotos(id, streamSel.value);
      toast('Camera frames saved');
      route();
    } catch (e) {
      msg.textContent = errText(e);
    }
  };

  render(
    h('div', { class: 'row' }, h('h1', {}, personName(p)), badge(p.status), h('span', { class: 'spacer' }), h('span', { class: 'muted' }, `version ${p.version}, updated ${fmtTime(p.updatedAt)}`)),
    h('div', { class: 'card' }, form.node, h('div', { class: 'row', style: 'margin-top:12px' }, h('button', { class: 'primary', onclick: save }, 'Save'), h('button', { onclick: async () => {
      await api.patchPerson(id, { status: p.status === 'active' ? 'disabled' : 'active' }, p.version).catch((e) => (msg.textContent = errText(e)));
      route();
    } }, p.status === 'active' ? 'Disable' : 'Enable'), msg)),
    h('h2', {}, `Reference photos (${p.photos.length})`),
    h(
      'div',
      { class: 'photos' },
      p.photos.map((ph) =>
        h(
          'div',
          { class: 'photo card', style: 'padding:8px' },
          authedImg(`/persons/${id}/photos/${ph.id}?variant=original`),
          h('div', { class: 'muted', style: 'font-size:11px;margin:6px 0' }, `${ph.source === 'camera' ? 'camera' : 'upload'} · size ${Math.round(ph.quality.faceSize)} · sharpness ${Math.round(ph.quality.sharpness)}`),
          h('button', { class: 'danger', onclick: async () => {
            try {
              await api.deletePhoto(id, ph.id);
              route();
            } catch (e) {
              toast(errText(e));
            }
          } }, 'Delete'),
        ),
      ),
    ),
    h('div', { class: 'card row', style: 'margin-top:12px' }, fileInput, h('button', { onclick: () => addPhotos() }, 'Add photos'), h('span', { class: 'spacer' }), streams.length ? [streamSel, h('button', { onclick: capture }, 'Capture from camera')] : h('span', { class: 'muted' }, 'no running streams to capture from')),
    h('h2', {}, 'Subject data'),
    h(
      'div',
      { class: 'card row' },
      h('button', { onclick: async () => {
        const u = await authedBlobUrl(`/persons/${id}/export`);
        h('a', { href: u, download: `person-${id}.zip` }).click();
      } }, 'Export (ZIP)'),
      h('span', { class: 'spacer' }),
      h('label', { class: 'row' }, purge, 'also delete journal and snapshots'),
      h('button', { class: 'danger', onclick: async () => {
        if (!confirm(`Permanently delete ${personName(p)}?`)) return;
        await api.deletePerson(id, purge.checked);
        toast('Deleted');
        location.hash = '#/persons';
      } }, 'Delete person'),
    ),
  );
}

/**
 * Live overlay preview of a stream: the next (downscaled) snapshot is requested as soon as the previous
 * one is displayed, so the preview runs at roughly the stream's processing rate. Stops when the page is left.
 */
function livePreview(streamId: () => string, width = 640): HTMLImageElement {
  const img = h('img', { class: 'live', alt: 'camera' });
  let stop = false;
  const tick = async () => {
    if (stop) return;
    try {
      const u = await authedBlobUrl(`/streams/${streamId()}/snapshot?overlay=true&width=${width}`);
      const old = img.src;
      img.src = u;
      await img.decode().catch(() => undefined);
      if (old) URL.revokeObjectURL(old);
      setTimeout(tick, 30);
    } catch {
      setTimeout(tick, 500); // no frame yet / stream restarting
    }
  };
  void tick();
  cleanup.push(() => (stop = true));
  return img;
}

/** Running streams, or a hint/start buttons when none is running. */
async function runningStreamsOrHint(): Promise<{ running: StreamInfo[]; hint: Node | null }> {
  const all = await api.listStreams();
  const running = all.filter((s) => s.status === 'running');
  if (running.length) return { running, hint: null };
  const stopped = all.filter((s) => s.status !== 'running');
  return {
    running,
    hint: h(
      'div',
      {},
      h('div', { class: 'muted', style: 'margin:8px 0' }, 'No running streams.'),
      stopped.length
        ? h('div', { class: 'row' }, stopped.map((s) => h('button', { onclick: async () => {
            await api.streamAction(s.id, 'start').catch((e) => toast(errText(e)));
            setTimeout(route, 1500);
          } }, `Start "${s.name}"`)))
        : h('a', { href: '#/streams' }, 'Add a webcam stream →'),
    ),
  };
}

const REASON_LABELS: Record<string, string> = {
  detScore: 'face is poorly visible',
  faceSize: 'face too far / too small',
  interocular: 'face too far',
  yaw: 'head turned — look at the camera',
  roll: 'head tilted',
  sharpness: 'frame blurred — hold still',
  brightness: 'poor lighting',
  inFrame: 'face at the edge of the frame',
};
const STATUS_LABELS: Record<string, string> = { match: 'Recognized', unknown: 'Unknown', uncertain: 'Uncertain', low_quality: 'Low quality', spoof: 'Spoof' };

async function recognizePage() {
  const { running, hint } = await runningStreamsOrHint();
  if (!running.length) return render(h('h1', {}, 'Recognition'), h('div', { class: 'card' }, hint));
  const sel = h('select', {}, running.map((s) => h('option', { value: s.id }, s.name)));
  const live = livePreview(() => sel.value);
  const result = h('div', { class: 'card' }, h('div', { class: 'muted' }, 'Stand in front of the camera and press "Recognize" (or Space).'));
  const history = h('tbody');
  let busy = false;

  const run = async () => {
    if (busy) return;
    busy = true;
    btn.disabled = true;
    result.replaceChildren(h('div', { class: 'verdict' }, 'Recognizing…'), h('div', { class: 'muted' }, 'Look at the camera and hold still for ~1 second'));
    const t0 = performance.now();
    try {
      const r = await api.recognizeOnStream(sel.value);
      const e = r.event;
      const person = e.person ? await api.getPerson(e.person.id).catch(() => null) : null;
      const top = r.candidates[0];
      const refPhoto = e.person ? top?.photoId : undefined;
      result.replaceChildren(
        h('div', { class: `verdict ${e.status}` }, STATUS_LABELS[e.status] ?? e.status),
        person
          ? h('div', {}, h('div', { style: 'font-size:20px;font-weight:600' }, personName(person)), h('div', { class: 'muted' }, [person.externalId, person.department, person.position].filter(Boolean).join(' · ')), h('a', { href: `#/person/${person.id}` }, 'profile →'))
          : null,
        h(
          'div',
          { class: 'faces' },
          h('figure', {}, h('img', { src: `data:image/png;base64,${r.face}` }), h('figcaption', {}, 'camera')),
          refPhoto && person ? h('figure', {}, authedImg(`/persons/${person.id}/photos/${refPhoto}?variant=aligned`), h('figcaption', {}, 'reference')) : null,
        ),
        h(
          'dl',
          { class: 'kv' },
          h('dt', {}, 'Score / 2nd'),
          h('dd', {}, `${e.score ?? '—'} / ${e.secondScore ?? '—'}`),
          h('dt', {}, 'Liveness'),
          h('dd', {}, e.liveness ? `${e.liveness.score} (${e.liveness.live ? 'live' : 'spoof'})` : 'off'),
          h('dt', {}, 'Frame agreement'),
          h('dd', {}, e.frameAgreement ?? '—'),
          h('dt', {}, 'Frames'),
          h('dd', {}, `${e.framesUsed} best of ${r.goodFrames} good (${r.framesSeen} total)`),
          h('dt', {}, 'Time'),
          h('dd', {}, `${Math.round(performance.now() - t0)} ms`),
        ),
        e.status === 'spoof'
          ? h('div', { class: 'err' }, 'Not a live face: a photo or a screen was shown to the camera. Identification was skipped.')
          : r.candidates.length
          ? h(
              'table',
              { style: 'margin-top:12px' },
              h('thead', {}, h('tr', {}, h('th', {}, 'Candidate'), h('th', {}, 'Score'), h('th', {}, ''))),
              h(
                'tbody',
                {},
                r.candidates.map((c) =>
                  h('tr', {}, h('td', {}, h('a', { href: `#/person/${c.personId}` }, c.name)), h('td', {}, c.score), h('td', {}, h('div', { class: 'bar' }, h('i', { style: `width:${Math.max(0, Math.min(1, c.score)) * 100}%` })))),
                ),
              ),
            )
          : h('div', { class: 'muted' }, 'Gallery is empty — enroll persons first.'),
      );
      history.prepend(h('tr', {}, h('td', {}, new Date().toLocaleTimeString('en-US')), h('td', {}, badge(e.status, STATUS_LABELS[e.status])), h('td', {}, person ? personName(person) : '—'), h('td', {}, e.score ?? '—')));
    } catch (e) {
      let text = errText(e);
      if (e instanceof ApiClientError && e.code === 'NO_FACE') text = 'No face found — stand in front of the camera';
      if (e instanceof ApiClientError && e.code === 'LOW_QUALITY') {
        text = `Not enough good-quality frames: ${((e.details.reasons as string[]) ?? []).map((x) => REASON_LABELS[x] ?? x).join('; ')}`;
      }
      result.replaceChildren(h('div', { class: 'verdict low_quality' }, 'Failed'), h('div', {}, text));
    } finally {
      busy = false;
      btn.disabled = false;
    }
  };
  const btn = h('button', { class: 'primary big-btn', onclick: run }, 'Recognize');
  const onKey = (ev: KeyboardEvent) => {
    if (ev.code === 'Space' && (ev.target as HTMLElement).tagName !== 'INPUT') {
      ev.preventDefault();
      void run();
    }
  };
  document.addEventListener('keydown', onKey);
  cleanup.push(() => document.removeEventListener('keydown', onKey));

  render(
    h('h1', {}, 'Recognition'),
    h('div', { class: 'row', style: 'margin-bottom:14px' }, sel, btn),
    h('div', { class: 'split' }, h('div', {}, live), h('div', {}, result, h('h2', {}, 'This session'), h('table', {}, history))),
  );
}

interface PendingPhoto {
  blob: Blob;
  name: string;
  source: 'upload' | 'camera';
  url: string;
}

/** Live camera panel: preview of a running stream + "take photo" (engine picks the best enrollment-quality frames). */
async function cameraPanel(onShots: (shots: PendingPhoto[]) => void): Promise<Node> {
  const { running, hint } = await runningStreamsOrHint();
  const box = h('div', { class: 'card' }, h('b', {}, 'Camera snapshot'));
  if (!running.length) {
    box.append(hint!);
    return box;
  }
  const sel = h('select', {}, running.map((s) => h('option', { value: s.id }, s.name)));
  const live = livePreview(() => sel.value);
  const status = h('span', { class: 'muted' });
  const shoot = h('button', { class: 'primary', onclick: async () => {
    shoot.disabled = true;
    status.textContent = 'Look at the camera… (one person in frame)';
    try {
      const r = await api.captureFromStream(sel.value);
      const shots = r.frames.map((f, i) => {
        const bytes = Uint8Array.from(atob(f.jpeg), (c) => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: 'image/jpeg' });
        return { blob, name: `camera-${Date.now()}-${i}.jpg`, source: 'camera' as const, url: URL.createObjectURL(blob) };
      });
      onShots(shots);
      status.textContent = `Frames added: ${shots.length}`;
    } catch (e) {
      status.replaceChildren(h('span', { class: 'err' }, errText(e)));
    } finally {
      shoot.disabled = false;
    }
  } }, 'Capture from camera');
  box.append(h('div', { class: 'row', style: 'margin:8px 0' }, sel, shoot, status), h('div', { style: 'max-width:480px' }, live));
  return box;
}

async function enroll() {
  const form = personForm();
  const consent = h('input', { type: 'checkbox' });
  const basis = h('input', { placeholder: 'e.g. written consent #123', style: 'width:320px' });
  const files = h('input', { type: 'file', accept: 'image/jpeg,image/png,image/webp', multiple: true });
  const previews = h('div', { class: 'photos' });
  const msg = h('div', {});
  let photos: PendingPhoto[] = [];
  let faceBox: number[] | undefined;

  const redraw = () =>
    previews.replaceChildren(
      ...photos.map((p, i) =>
        h(
          'div',
          { class: 'photo' },
          h('img', { src: p.url }),
          h('div', { class: 'row', style: 'margin-top:4px' }, h('span', { class: 'muted' }, p.source === 'camera' ? 'camera' : 'file'), h('span', { class: 'spacer' }), h('button', { class: 'danger', onclick: () => {
            URL.revokeObjectURL(p.url);
            photos.splice(i, 1);
            faceBox = undefined;
            redraw();
          } }, '✕')),
        ),
      ),
    );
  const add = (list: PendingPhoto[]) => {
    photos = [...photos, ...list];
    faceBox = undefined;
    redraw();
  };
  files.addEventListener('change', () => {
    add([...(files.files ?? [])].map((f) => ({ blob: f, name: f.name, source: 'upload' as const, url: URL.createObjectURL(f) })));
    files.value = '';
  });

  const submit = async (allowDuplicate = false) => {
    if (!photos.length) return msg.replaceChildren(h('span', { class: 'err' }, 'Add at least one photo'));
    msg.replaceChildren('Processing…');
    const data = { ...form.values(false), consent: { obtained: consent.checked, basis: basis.value || undefined } };
    const fd = new FormData();
    fd.append('data', JSON.stringify(data));
    if (faceBox) fd.append('faceBox', JSON.stringify(faceBox));
    fd.append('photoSources', JSON.stringify(photos.map((p) => p.source)));
    for (const p of photos) fd.append('photos', p.blob, p.name);
    try {
      const p = await api.request<Person>('POST', `/persons${allowDuplicate ? '?allowDuplicate=true' : ''}`, fd);
      toast(`Enrolled: ${personName(p)}`);
      location.hash = `#/person/${p.id}`;
    } catch (e) {
      if (!(e instanceof ApiClientError)) return msg.replaceChildren(h('span', { class: 'err' }, errText(e)));
      const idx = (e.details.photoIndex as number | undefined) ?? 0;
      const out: Child[] = [h('div', { class: 'err' }, `Photo #${idx + 1}: ${errText(e)}`)];
      if (e.code === 'DUPLICATE_SUSPECTED') {
        const c = e.details.candidate as { personId: string; name?: string; score: number };
        out.push(
          h('div', { class: 'row', style: 'margin-top:8px' }, `Similar to: `, h('a', { href: `#/person/${c.personId}` }, c.name ?? c.personId), ` (score ${c.score})`, h('button', { onclick: () => submit(true) }, 'This is a different person — enroll')),
        );
      }
      if (e.code === 'MULTIPLE_FACES' && photos[idx]) {
        const boxes = e.details.boxes as number[][];
        const img = h('img', { src: photos[idx].url });
        const wrap = h('div', { class: 'preview' }, img);
        img.onload = () => {
          const sx = img.clientWidth / img.naturalWidth;
          const sy = img.clientHeight / img.naturalHeight;
          for (const b of boxes) {
            wrap.append(
              h('div', {
                class: 'fbox',
                title: 'Select this face',
                style: `left:${b[0] * sx}px;top:${b[1] * sy}px;width:${b[2] * sx}px;height:${b[3] * sy}px`,
                onclick: () => {
                  // faceBox applies to the first photo: move the chosen one to the front.
                  photos = [photos[idx], ...photos.filter((_, j) => j !== idx)];
                  redraw();
                  faceBox = b;
                  void submit(allowDuplicate);
                },
              }),
            );
          }
        };
        out.push(h('div', { class: 'muted', style: 'margin:8px 0' }, 'Click the correct face:'), wrap);
      }
      msg.replaceChildren(...(out.filter(Boolean) as Node[]));
    }
  };

  render(
    h('h1', {}, 'Enrollment'),
    h('div', { class: 'card' }, form.node),
    h('div', { class: 'card' }, h('label', { class: 'row' }, consent, h('b', {}, 'Consent to biometric data processing obtained')), h('div', { class: 'row', style: 'margin-top:8px' }, 'Legal basis:', basis)),
    h(
      'div',
      { class: 'card' },
      h('div', { class: 'muted', style: 'margin-bottom:8px' }, '3–5 photos recommended, at least one from the target camera. Files and camera snapshots can be combined.'),
      h('div', { class: 'row' }, h('b', {}, 'Files:'), files),
    ),
    await cameraPanel(add),
    h('h2', {}, 'Enrollment photos'),
    previews,
    h('div', { class: 'row', style: 'margin-top:12px' }, h('button', { class: 'primary', onclick: () => submit() }, 'Enroll')),
    h('div', { class: 'card', style: 'margin-top:12px' }, msg),
  );
  redraw();
}

async function streamsPage() {
  const list = await api.listStreams();
  let cams: { deviceId: string; label: string }[] = [];
  try {
    cams = await api.cameras();
  } catch {
    /* ignore */
  }
  const name = h('input', { placeholder: 'Name', value: 'Entrance' });
  const type = h('select', {}, h('option', { value: 'webcam' }, 'Webcam'), h('option', { value: 'rtsp' }, 'RTSP / IP camera'), h('option', { value: 'file' }, 'Video file (debug)'));
  const device = h('select', {}, cams.map((c) => h('option', { value: c.deviceId }, c.label)));
  const url = h('input', { placeholder: 'rtsp://user:pass@host/stream or file path', style: 'width:360px' });
  const fps = h('input', { type: 'number', value: 10, min: 1, max: 30, style: 'width:70px' });
  const loop = h('input', { type: 'checkbox' });
  const msg = h('span', { class: 'err' });
  const sync = () => {
    device.style.display = type.value === 'webcam' ? '' : 'none';
    url.style.display = type.value === 'webcam' ? 'none' : '';
  };
  type.addEventListener('change', sync);
  sync();

  const row = (s: StreamInfo) =>
    h(
      'tr',
      {},
      h('td', {}, h('b', {}, s.name), h('div', { class: 'muted' }, s.type === 'webcam' ? cams.find((c) => c.deviceId === s.deviceId)?.label ?? 'webcam' : s.url ?? '')),
      h('td', {}, badge(s.status), s.statusMessage ? h('div', { class: 'muted' }, s.statusMessage) : ''),
      h('td', {}, s.fps),
      h('td', {}, s.droppedFrames),
      h('td', {}, s.activeTracks),
      h(
        'td',
        {},
        h('div', { class: 'row' },
          s.status === 'stopped' || s.status === 'error'
            ? h('button', { onclick: async () => { await api.streamAction(s.id, 'start').catch((e) => toast(errText(e))); route(); } }, 'Start')
            : h('button', { onclick: async () => { await api.streamAction(s.id, 'stop'); route(); } }, 'Stop'),
          h('button', { class: 'danger', onclick: async () => { if (confirm('Delete stream?')) { await api.deleteStream(s.id); route(); } } }, 'Delete'),
        ),
      ),
    );

  render(
    h('h1', {}, 'Streams'),
    h('table', {}, h('thead', {}, h('tr', {}, h('th', {}, 'Stream'), h('th', {}, 'Status'), h('th', {}, 'FPS'), h('th', {}, 'Dropped'), h('th', {}, 'Tracks'), h('th', {}, ''))), h('tbody', {}, list.map(row))),
    h('h2', {}, 'Add stream'),
    h(
      'div',
      { class: 'card row' },
      name,
      type,
      device,
      url,
      h('label', { class: 'row' }, 'FPS', fps),
      h('label', { class: 'row' }, loop, 'loop file'),
      h('button', { class: 'primary', onclick: async () => {
        try {
          await api.createStream({
            name: name.value,
            type: type.value as 'webcam',
            deviceId: type.value === 'webcam' ? device.value : undefined,
            url: type.value !== 'webcam' ? url.value : undefined,
            detectFps: Number(fps.value),
            enabled: true,
            loop: loop.checked || undefined,
          });
          route();
        } catch (e) {
          msg.textContent = errText(e);
        }
      } }, 'Add and start'),
      msg,
    ),
  );
  const t = setInterval(async () => {
    const fresh = await api.listStreams().catch(() => null);
    const tbody = view().querySelector('tbody');
    if (fresh && tbody) tbody.replaceChildren(...fresh.map(row));
  }, 2000);
  cleanup.push(() => clearInterval(t));
}

async function eventsPage(params: URLSearchParams) {
  const status = params.get('status') ?? '';
  const j = await api.recognitions(Object.fromEntries([['limit', '100'], ...(status ? [['status', status]] : []), ...(params.get('cursor') ? [['cursor', params.get('cursor')!]] : [])]));
  const sel = h('select', { onchange: () => (location.hash = `#/events?status=${sel.value}`) }, ['', 'match', 'unknown', 'uncertain', 'low_quality', 'spoof'].map((s) => h('option', { value: s, selected: s === status }, s || 'all')));
  const names = new Map<string, string>();
  for (const pid of new Set(j.items.map((e) => e.personId).filter(Boolean))) {
    const p = await api.getPerson(pid).catch(() => null);
    if (p) names.set(pid, personName(p));
  }
  render(
    h('div', { class: 'row' }, h('h1', {}, 'Recognition journal'), h('span', { class: 'spacer' }), sel),
    h(
      'table',
      {},
      h('thead', {}, h('tr', {}, h('th', {}, 'Time'), h('th', {}, 'Stream'), h('th', {}, 'Track'), h('th', {}, 'Status'), h('th', {}, 'Person'), h('th', {}, 'Score / 2nd'), h('th', {}, 'Liveness'), h('th', {}, 'Frames'), h('th', {}, 'Attempt'), h('th', {}, 'Latency'))),
      h(
        'tbody',
        {},
        j.items.map((e) =>
          h(
            'tr',
            {},
            h('td', {}, fmtTime(e.ts)),
            h('td', {}, e.sourceId.slice(-6)),
            h('td', {}, e.trackId),
            h('td', {}, badge(e.status)),
            h('td', {}, e.personId ? h('a', { href: `#/person/${e.personId}` }, names.get(e.personId) ?? e.personId.slice(0, 8)) : '—'),
            h('td', {}, `${e.score ?? '—'} / ${e.secondScore ?? '—'}`),
            h('td', {}, e.liveness?.score ?? '—'),
            h('td', {}, e.framesUsed),
            h('td', {}, e.attempt),
            h('td', {}, `${e.latencyMs} ms`),
          ),
        ),
      ),
    ),
    j.nextCursor ? h('div', { class: 'row', style: 'margin-top:10px' }, h('button', { onclick: () => (location.hash = `#/events?status=${status}&cursor=${j.nextCursor}`) }, 'Older →')) : null,
  );
}

const SETTINGS: [string, string, string][] = [
  ['match.acceptThreshold', 'Match threshold (accept)', 'number'],
  ['match.rejectThreshold', 'Reject threshold', 'number'],
  ['match.margin', 'Margin top1 − top2', 'number'],
  ['match.minFrameAgreement', 'Frame agreement', 'number'],
  ['quality.minFaceSize', 'Min face size, px', 'number'],
  ['quality.maxYaw', 'Max yaw', 'number'],
  ['quality.minSharpness', 'Min sharpness', 'number'],
  ['pipeline.detectFps', 'Detection FPS', 'number'],
  ['pipeline.cooldownMs', 'Cooldown, ms', 'number'],
  ['pipeline.maxConcurrentTracks', 'Concurrent tracks', 'number'],
  ['burst.maxFrames', 'Frames per burst', 'number'],
  ['burst.topK', 'Top frames', 'number'],
  ['liveness.enabled', 'Anti-spoofing (reject photos / screens)', 'checkbox'],
  ['liveness.threshold', 'Liveness threshold (0..1)', 'number'],
  ['events.storeSnapshots', 'Store event snapshots', 'checkbox'],
  ['events.snapshotRetentionDays', 'Snapshot retention, days', 'number'],
  ['events.logRetentionDays', 'Journal retention, days', 'number'],
  ['models.embedder', 'Embedding model (restart)', 'text'],
  ['models.executionProvider', 'Accelerator: auto/cpu/cuda/dml (restart)', 'text'],
];

async function settings() {
  const { config, restartRequired } = await api.getConfig();
  const get = (p: string) => p.split('.').reduce((o: any, k) => o?.[k], config);
  const inputs = SETTINGS.map(([p, label, type]) => {
    const el = h('input', type === 'checkbox' ? { type, checked: !!get(p) } : { type, value: get(p), step: 'any' });
    return { p, type, el, node: h('label', { class: 'f' }, h('span', {}, label), el) };
  });
  const msg = h('div', {});
  const save = async () => {
    const patch: Record<string, any> = {};
    for (const i of inputs) {
      const v = i.type === 'checkbox' ? i.el.checked : i.type === 'number' ? Number(i.el.value) : i.el.value;
      if (v === get(i.p)) continue;
      const keys = i.p.split('.');
      let o = patch;
      keys.slice(0, -1).forEach((k) => (o = o[k] ??= {}));
      o[keys.at(-1)!] = v;
    }
    try {
      const r = await api.patchConfig(patch);
      msg.replaceChildren(h('span', {}, 'Saved. '), r.restartRequired.length ? h('span', { class: 'err' }, `Engine restart required: ${r.restartRequired.join(', ')}`) : '');
    } catch (e) {
      msg.replaceChildren(h('span', { class: 'err' }, errText(e)));
    }
  };
  const jobMsg = h('span', { class: 'muted' });
  const reindex = async () => {
    const { jobId } = await api.request<{ jobId: string }>('POST', '/admin/reindex');
    const poll = async () => {
      const j = await api.job(jobId);
      jobMsg.textContent = `${j.state}: ${j.done}/${j.total}${j.errors.length ? `, errors ${j.errors.length}` : ''}`;
      if (j.state === 'running') setTimeout(poll, 700);
    };
    void poll();
  };
  const tokenBox = h('code', {}, '••••••••');
  render(
    h('h1', {}, 'Settings'),
    restartRequired.length ? h('div', { class: 'card err' }, `Pending restart: ${restartRequired.join(', ')}`) : null,
    h('div', { class: 'card' }, h('div', { class: 'form' }, inputs.map((i) => i.node)), h('div', { class: 'row', style: 'margin-top:12px' }, h('button', { class: 'primary', onclick: save }, 'Save'), msg)),
    h('div', { class: 'muted', style: 'margin:-6px 0 14px' }, 'Default thresholds are placeholders. Production values come from calibration on the target camera: npm run calibrate.'),
    h('h2', {}, 'Index'),
    h('div', { class: 'card row' }, h('button', { onclick: reindex }, 'Reindex all photos'), jobMsg),
    h('h2', {}, 'API access'),
    h(
      'div',
      { class: 'card' },
      h('div', { class: 'row' }, 'Address: ', h('code', {}, conn!.baseUrl)),
      h('div', { class: 'row', style: 'margin-top:8px' }, 'Token: ', tokenBox, h('button', { onclick: () => (tokenBox.textContent = conn!.token) }, 'Show'), h('button', { class: 'danger', onclick: async () => {
        if (!confirm('Rotate the token? External integrations will stop working until they are updated with the new token.')) return;
        const r = await api.request<{ token: string }>('POST', '/admin/token/rotate');
        if (webMode) {
          writeToken(r.token);
          setConn({ ...conn!, token: r.token });
        }
        toast('Token rotated');
      } }, 'Rotate')),
    ),
    h('h2', {}, 'Full configuration'),
    h('pre', {}, JSON.stringify(config, null, 2)),
  );
}

// ---------- router ----------

async function route() {
  cleanup.forEach((fn) => fn());
  cleanup = [];
  const [pathPart, query = ''] = (location.hash.slice(1) || '/dashboard').split('?');
  const parts = pathPart.split('/').filter(Boolean);
  const params = new URLSearchParams(query);
  document.querySelectorAll('.nav a').forEach((a) => a.classList.toggle('active', (a as HTMLElement).dataset.route === (parts[0] === 'person' ? 'persons' : parts[0])));
  if (!conn) {
    if (webMode) return; // the login form is shown instead
    return render(h('h1', {}, 'Engine is starting…'), h('p', { class: 'muted' }, 'Loading models and gallery.'));
  }
  try {
    switch (parts[0]) {
      case 'persons':
        return await persons(params);
      case 'person':
        return await personPage(parts[1]);
      case 'enroll':
        return await enroll();
      case 'recognize':
        return await recognizePage();
      case 'streams':
        return await streamsPage();
      case 'events':
        return await eventsPage(params);
      case 'settings':
        return await settings();
      default:
        return await dashboard();
    }
  } catch (e) {
    if (webMode && e instanceof ApiClientError && e.status === 401) {
      writeToken(null);
      conn = null;
      location.reload();
      return;
    }
    render(h('h1', {}, 'Error'), h('p', { class: 'err' }, errText(e)));
  }
}

// ---------- web mode (no Electron): same-origin API, token typed by the operator ----------

const TOKEN_KEY = 'faceid.token';

function readToken(): string | null {
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

function writeToken(t: string | null) {
  try {
    if (t) sessionStorage.setItem(TOKEN_KEY, t);
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable: token lives only in memory */
  }
}

function webLogin(onConn: (c: Conn) => void, error = '') {
  const input = h('input', { type: 'password', placeholder: 'API token', style: 'width:420px', autofocus: true });
  const msg = h('div', { class: 'err' }, error);
  const submit = async (e: Event) => {
    e.preventDefault();
    const c = { baseUrl: `${location.origin}/api/v1`, token: input.value.trim() };
    const r = await fetch(`${c.baseUrl}/health`, { headers: { Authorization: `Bearer ${c.token}` } }).catch(() => null);
    if (!r?.ok) {
      msg.textContent = r?.status === 401 ? 'Invalid token' : 'Engine unavailable';
      return;
    }
    writeToken(c.token);
    onConn(c);
  };
  render(
    h('h1', {}, 'Sign in'),
    h('p', { class: 'muted' }, 'The token is printed to the console when the engine starts (the "API token" line).'),
    h('form', { class: 'row', onsubmit: submit }, input, h('button', { class: 'primary' }, 'Sign in')),
    msg,
  );
  document.getElementById('engine-state')!.textContent = 'not signed in';
}

if (webMode) {
  let connCb: (c: Conn) => void = () => undefined;
  window.faceid = {
    getConn: async () => {
      const t = readToken();
      return t ? { baseUrl: `${location.origin}/api/v1`, token: t } : null;
    },
    getStatus: async () => ({ status: 'web', error: '' }),
    onConn: (cb) => (connCb = cb),
    onStatus: () => undefined,
    submitPassword: () => undefined,
  };
  // Session expired / token rotated elsewhere -> back to the login form.
  window.addEventListener('unhandledrejection', (e) => {
    if (e.reason instanceof ApiClientError && e.reason.status === 401) {
      writeToken(null);
      webLogin(connCb, 'Session expired, please sign in again');
    }
  });
  const foot = document.querySelector('.nav-foot')!;
  foot.after(h('a', { href: '#', onclick: (e: Event) => { e.preventDefault(); writeToken(null); conn = null; webLogin(connCb); } }, 'Sign out'));
  void window.faceid.getConn().then((c) => {
    if (!c) webLogin((cc) => connCb(cc));
  });
}

window.addEventListener('hashchange', route);
window.faceid.onConn((c) => setConn(c));
window.faceid.onStatus((s) => {
  document.getElementById('engine-state')!.textContent = `engine: ${s.status}${s.error ? ` — ${s.error}` : ''}`;
});
void window.faceid.getConn().then((c) => {
  setConn(c);
  if (!c) void route();
});
