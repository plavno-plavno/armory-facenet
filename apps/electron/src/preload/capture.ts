// Preload for the hidden capture window: hands the engine MessagePort to the page.
import { ipcRenderer } from 'electron';

ipcRenderer.on('capture:port', (e) => {
  window.postMessage('capture:port', '*', e.ports);
});
