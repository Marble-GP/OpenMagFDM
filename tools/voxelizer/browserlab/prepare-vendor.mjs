import { mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const directory = fileURLToPath(new URL('./dist/vendor/', import.meta.url));
await mkdir(directory, { recursive: true });
for (const [source, destination] of [
  ['build/three.module.js', 'three.module.js'],
  ['build/three.core.js', 'three.core.js'],
  ['examples/jsm/controls/OrbitControls.js', 'OrbitControls.js'],
  ['LICENSE', 'THREE-LICENSE.txt'],
]) {
  await copyFile(new URL(`./node_modules/three/${source}`, import.meta.url), `${directory}/${destination}`);
}
console.log('Prepared local Three.js 0.186.1 assets (no runtime CDN).');
