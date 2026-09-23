// Hidden capture window (spec §7.2 webcam): getUserMedia -> MediaStreamTrackProcessor -> RGBA frames
// throttled to the stream's detectFps -> MessagePort -> engine (which converts to BGR24).

type Msg =
  | { type: 'list'; reqId: number }
  | { type: 'open'; streamId: string; deviceId?: string; width?: number; height?: number; fps: number }
  | { type: 'close'; streamId: string };

interface Capture {
  stream: MediaStream;
  reader: ReadableStreamDefaultReader<VideoFrame>;
  stopped: boolean;
}

declare class MediaStreamTrackProcessor {
  constructor(init: { track: MediaStreamTrack });
  readable: ReadableStream<VideoFrame>;
}

let port: MessagePort | null = null;
const captures = new Map<string, Capture>();

function send(msg: unknown, transfer: Transferable[] = []): void {
  port?.postMessage(msg, transfer);
}

function stop(id: string): void {
  const c = captures.get(id);
  if (!c) return;
  c.stopped = true;
  c.reader.cancel().catch(() => undefined);
  c.stream.getTracks().forEach((t) => t.stop());
  captures.delete(id);
}

async function listDevices(reqId: number): Promise<void> {
  let devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
  if (devices.some((d) => !d.label) && captures.size === 0) {
    // Labels are only exposed after a permission grant; open and immediately release a camera.
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      s.getTracks().forEach((t) => t.stop());
      devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    } catch {
      /* no camera or denied */
    }
  }
  send({ type: 'devices', reqId, devices: devices.map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` })) });
}

async function open(m: Extract<Msg, { type: 'open' }>): Promise<void> {
  stop(m.streamId);
  send({ type: 'status', streamId: m.streamId, state: 'starting' });
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        deviceId: m.deviceId ? { exact: m.deviceId } : undefined,
        width: m.width ? { ideal: m.width } : { ideal: 1280 },
        height: m.height ? { ideal: m.height } : { ideal: 720 },
        frameRate: { ideal: 30 },
      },
    });
  } catch (e) {
    send({ type: 'status', streamId: m.streamId, state: 'error', message: (e as Error).message });
    return;
  }
  const track = stream.getVideoTracks()[0];
  const reader = new MediaStreamTrackProcessor({ track }).readable.getReader();
  const cap: Capture = { stream, reader, stopped: false };
  captures.set(m.streamId, cap);
  track.addEventListener('ended', () => {
    if (!cap.stopped) send({ type: 'status', streamId: m.streamId, state: 'error', message: 'camera disconnected' });
  });
  const interval = 1000 / m.fps;
  let last = 0;
  let canvas: OffscreenCanvas | null = null;
  let ctx: OffscreenCanvasRenderingContext2D | null = null;
  for (;;) {
    const { value: frame, done } = await reader.read().catch(() => ({ value: undefined, done: true }));
    if (done || !frame || cap.stopped) {
      frame?.close();
      break;
    }
    const now = performance.now();
    if (now - last >= interval) {
      last = now;
      const w = frame.displayWidth;
      const h = frame.displayHeight;
      if (!canvas || canvas.width !== w || canvas.height !== h) {
        canvas = new OffscreenCanvas(w, h);
        ctx = canvas.getContext('2d', { willReadFrequently: true });
      }
      ctx!.drawImage(frame, 0, 0);
      const img = ctx!.getImageData(0, 0, w, h);
      send({ type: 'frame', streamId: m.streamId, width: w, height: h, data: img.data.buffer }, [img.data.buffer]);
    }
    frame.close();
  }
  if (!cap.stopped) stop(m.streamId);
  send({ type: 'status', streamId: m.streamId, state: 'stopped' });
}

window.addEventListener('message', (e) => {
  if (e.data !== 'capture:port' || !e.ports[0]) return;
  // A new engine instance (e.g. after a watchdog restart): drop old captures, it will reopen them.
  for (const id of [...captures.keys()]) stop(id);
  port?.close();
  port = e.ports[0];
  port.onmessage = (ev: MessageEvent<Msg>) => {
    const m = ev.data;
    if (m.type === 'list') void listDevices(m.reqId);
    else if (m.type === 'open') void open(m);
    else if (m.type === 'close') stop(m.streamId);
  };
  port.start();
});
