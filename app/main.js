// Electron main process: the host. It starts the agent as a child process
// (`computeruse sidecar`), speaks JSON-RPC to it over stdin/stdout, and relays
// between the sidecar and the page. The page never touches Node or the
// sidecar directly: it only gets the two calls in preload.js.
//
//   page --ipc--> main --stdin JSON line--> sidecar
//   page <--ipc-- main <--stdout JSON line-- sidecar
//
// npm start           real agent (moves the mouse)
// npm run demo        scripted steps, no mouse
// CU_SNAPSHOT=x.png   save a picture of the window and quit (self-check)

const { app, BrowserWindow, ipcMain, globalShortcut, screen } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const DEMO = process.argv.includes('--demo') || process.env.CU_DEMO === '1';
const REPO = path.join(__dirname, '..');
const SIDECAR = process.env.CU_SIDECAR || path.join(REPO, '.venv', 'Scripts', 'computeruse.exe');
const SNAPSHOT = process.env.CU_SNAPSHOT;

let win = null;
let sidecar = null;
let nextId = 1;
const pending = new Map(); // request id -> resolve

function toPage(msg) {
  if (win && !win.isDestroyed()) win.webContents.send('agent:event', msg);
}

function startSidecar() {
  // loop: our agent loop, router and grounder, with Claude (or CU_LOOP_BRAIN=groq) only deciding;
  // claude: Claude Code drives the MCP server; groq: the earlier desktop loop
  const brain = process.env.CU_BRAIN || 'loop';
  sidecar = spawn(SIDECAR, DEMO ? ['sidecar', '--demo'] : ['sidecar', '--brain', brain], { cwd: REPO, windowsHide: true });
  readline.createInterface({ input: sidecar.stdout }).on('line', (line) => {
    let msg;
    try {
      msg = JSON.parse(line);
    } catch {
      console.error('[sidecar sent a line that is not JSON]', line);
      return;
    }
    if (msg.id !== undefined && pending.has(msg.id)) {
      pending.get(msg.id)(msg); // a reply: hand it to whoever asked
      pending.delete(msg.id);
    } else if (msg.method) {
      toPage(msg); // a notification: step, started, finished
      if (msg.method !== 'screen') console.log(`[agent] ${msg.method} ${JSON.stringify(msg.params).slice(0, 300)}`);
    }
  });
  sidecar.stderr.on('data', (d) => process.stderr.write(`[sidecar] ${d}`));
  sidecar.on('error', (e) => toPage({ method: 'sidecar-exit', params: { error: String(e) } }));
  sidecar.on('exit', (code) => {
    for (const resolve of pending.values()) resolve({ error: { code: -32001, message: 'the agent process ended' } });
    pending.clear();
    toPage({ method: 'sidecar-exit', params: { code } });
  });
}

function request(method, params = {}) {
  return new Promise((resolve) => {
    if (!sidecar || sidecar.exitCode !== null || !sidecar.stdin.writable) {
      resolve({ error: { code: -32001, message: 'the agent process is not running' } });
      return;
    }
    const id = nextId++;
    pending.set(id, resolve);
    sidecar.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  });
}

ipcMain.handle('agent:request', (_event, method, params) => request(method, params));

function createWindow() {
  // CU_LAYOUT=split (for recording): this window on the right 36% of the screen; the sidecar puts the
  // app or the agent's Chrome on the left 64% (agent/runner.py place_target), so both are in frame.
  const split = process.env.CU_LAYOUT === 'split';
  const area = screen.getPrimaryDisplay().workArea;
  const right = Math.round(area.width * 0.64);
  win = new BrowserWindow({
    ...(split ? { x: area.x + right, y: area.y, width: area.width - right, height: area.height }
              : { width: 1280, height: 800 }),
    minWidth: split ? 480 : 900,
    minHeight: 600,
    backgroundColor: '#0b0d10',
    title: 'computer-use',
    autoHideMenuBar: true,
    // The self-check renders offscreen: a window opened from a background shell is not
    // always painted on screen, and capturing it then returns an empty image.
    show: !SNAPSHOT,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      offscreen: Boolean(SNAPSHOT),
    },
  });
  const query = { mode: DEMO ? 'demo' : 'live' };
  if (process.env.CU_AUTORUN) query.autorun = process.env.CU_AUTORUN;
  if (process.env.CU_SKY) query.sky = process.env.CU_SKY; // dev knob: which part of the room is behind the desk
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'), { query });

  if (SNAPSHOT) {
    const after = Number(process.env.CU_SNAPSHOT_AFTER || 4000);
    win.webContents.once('did-finish-load', () => {
      setTimeout(async () => {
        const image = await win.webContents.capturePage();
        if (image.isEmpty()) {
          console.error('[snapshot] the captured image is empty');
          app.exit(1);
          return;
        }
        fs.writeFileSync(SNAPSHOT, image.toPNG());
        app.quit();
      }, after);
    });
  }
}

app.whenReady().then(() => {
  startSidecar();
  createWindow();
  // works even while the agent has the focus in another app
  globalShortcut.register('Control+Shift+X', () => request('stop'));
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
  if (sidecar && sidecar.exitCode === null) sidecar.stdin.end(); // closing stdin makes the sidecar stop and exit
});

app.on('window-all-closed', () => app.quit());
