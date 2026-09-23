// Preload for the admin UI: exposes only the engine connection (URL + token) and the unlock channel.
import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('faceid', {
  getConn: () => ipcRenderer.invoke('engine:getConn'),
  getStatus: () => ipcRenderer.invoke('engine:getStatus'),
  onConn: (cb: (c: unknown) => void) => ipcRenderer.on('engine:conn', (_e, c) => cb(c)),
  onStatus: (cb: (s: unknown) => void) => ipcRenderer.on('engine:status', (_e, s) => cb(s)),
  submitPassword: (pw: string) => ipcRenderer.send('unlock:password', pw),
});
