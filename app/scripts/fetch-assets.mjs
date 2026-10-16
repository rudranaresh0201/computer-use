// Downloads the scene's CC0 assets from Poly Haven into app/assets/ph/ (gitignored).
// One environment photo for light and background, a few real props for the desk.
// Re-running skips files that are already there.
//
//   npm run assets

import { mkdir, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const OUT = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'assets', 'ph');
const HDRI = { id: 'hotel_room', res: '2k' };
const MODELS = ['metal_office_desk', 'modern_arm_chair_01', 'desk_lamp_arm_01', 'potted_plant_02', 'office_notepads'];
const RES = '1k';

async function exists(file) {
  try {
    return (await stat(file)).size > 0;
  } catch {
    return false;
  }
}

async function download(url, file) {
  if (await exists(file)) return false;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${url}`);
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, Buffer.from(await res.arrayBuffer()));
  return true;
}

async function files(id) {
  const res = await fetch(`https://api.polyhaven.com/files/${id}`);
  if (!res.ok) throw new Error(`Poly Haven has no asset ${id} (${res.status})`);
  return res.json();
}

const hdri = (await files(HDRI.id)).hdri[HDRI.res].hdr;
const got = await download(hdri.url, path.join(OUT, `${HDRI.id}.hdr`));
console.log(`${got ? 'downloaded' : 'have'} ${HDRI.id}.hdr`);

for (const id of MODELS) {
  const gltf = (await files(id)).gltf[RES].gltf;
  const dir = path.join(OUT, id);
  let n = Number(await download(gltf.url, path.join(dir, `${id}.gltf`)));
  for (const [rel, info] of Object.entries(gltf.include)) n += Number(await download(info.url, path.join(dir, rel)));
  console.log(`${n ? `downloaded ${n} files of` : 'have'} ${id}`);
}
