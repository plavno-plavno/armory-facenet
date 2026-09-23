// Bundles the Electron app into dist/: main (CJS), preloads (CJS), engine (ESM for utilityProcess),
// capture window and admin UI (browser IIFE).
import { build } from 'esbuild';
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';

const here = path.dirname(new URL(import.meta.url).pathname);
const out = path.join(here, 'dist');
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

const nativeExternals = ['electron', 'onnxruntime-node', 'sharp', 'pino', 'pino-roll', 'thread-stream', 'bufferutil', 'utf-8-validate'];
const common = { bundle: true, sourcemap: true, logLevel: 'warning', legalComments: 'none' };

await Promise.all([
  build({ ...common, entryPoints: [path.join(here, 'src/main/main.ts')], outfile: path.join(out, 'main.cjs'), platform: 'node', format: 'cjs', external: nativeExternals }),
  build({ ...common, entryPoints: [path.join(here, 'src/preload/admin.ts')], outfile: path.join(out, 'preload-admin.cjs'), platform: 'node', format: 'cjs', external: ['electron'] }),
  build({ ...common, entryPoints: [path.join(here, 'src/preload/capture.ts')], outfile: path.join(out, 'preload-capture.cjs'), platform: 'node', format: 'cjs', external: ['electron'] }),
  build({
    ...common,
    entryPoints: [path.join(here, '../engine/src/main.ts')],
    outfile: path.join(out, 'engine.mjs'),
    platform: 'node',
    format: 'esm',
    target: 'node22',
    external: nativeExternals,
    // CJS dependencies inside an ESM bundle need a real require().
    banner: { js: "import { createRequire as __cr } from 'node:module'; const require = __cr(import.meta.url);" },
  }),
  build({ ...common, entryPoints: [path.join(here, 'src/capture/capture.ts')], outfile: path.join(out, 'capture/capture.js'), platform: 'browser', format: 'iife', minify: true }),
  build({ ...common, entryPoints: [path.join(here, 'src/admin-ui/app.ts')], outfile: path.join(out, 'admin-ui/app.js'), platform: 'browser', format: 'iife', minify: true }),
  build({ ...common, entryPoints: [path.join(here, 'src/admin-ui/unlock.ts')], outfile: path.join(out, 'admin-ui/unlock.js'), platform: 'browser', format: 'iife' }),
]);
cpSync(path.join(here, 'src/capture/capture.html'), path.join(out, 'capture/capture.html'));
for (const f of ['index.html', 'unlock.html', 'styles.css']) cpSync(path.join(here, 'src/admin-ui', f), path.join(out, 'admin-ui', f));
console.log('built', out);
