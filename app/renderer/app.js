// The page: turns the agent's notifications into what you see.
//
//   planning -> status "planning", the goal, the robot thinks
//   plan     -> the timeline (the planner's checkpoints)
//   started  -> status "working", the clock starts, the robot walks to the desk and sits
//   thought  -> the reasoning panel types it out; the robot tilts its head
//   screen   -> the screen it sees, on the panel and on the 3D monitor
//   step     -> an action row under the current checkpoint, badged by the channel that found
//               its target (api / tree / eyes / vision); the marker jumps to where it pointed
//   receipt  -> the proof a checkpoint really happened (verified or contradicted)
//   check    -> the checkpoint is done; the robot nods
//   finished -> the result card; the robot stands up: thumbs up, or "no"
//
// Unticked checkpoints stay unticked at the end: the timeline shows what was verified,
// not what we hoped. window.agent comes from preload.js: request(method, params) and onEvent(cb).

import { Room, CHANNEL_COLOR } from './scene.js';

const $ = (id) => document.getElementById(id);
const room = new Room($('scene'));
const params = new URLSearchParams(location.search);
const DEMO = params.get('mode') === 'demo';

const TARGETS = {
  '': { label: 'Auto', icon: 'wand' },
  calculator: { label: 'Calculator', icon: 'calculator' },
  notepad_pp: { label: 'Notepad++', icon: 'file-text' },
  paint: { label: 'Paint', icon: 'paintbrush' },
  sheets: { label: 'Sheets', icon: 'file-spreadsheet' },
  docs: { label: 'Docs', icon: 'file-text' },
  gmail: { label: 'Gmail', icon: 'mail' },
  tinkercad: { label: 'Tinkercad', icon: 'box' },
  blender: { label: 'Blender', icon: 'layers' },
  excalidraw: { label: 'Excalidraw', icon: 'paintbrush' },
};
const EXAMPLES = [
  'Draw a house with a sun and a tree in Paint, then save it as house.png',
  'Make a sheet of this week\'s expenses with totals and a chart',
  'Design a keychain with my name in Tinkercad and export the STL',
];
const ACTION_ICON = { click: 'mouse-pointer-click', drag: 'mouse-pointer-click', type: 'keyboard', key: 'keyboard', locate: 'eye', scroll: 'mouse-pointer-click' };

let running = false;
let appLabel = '';
let target = '';
let plan = []; // [{text, done, actions: [], receipt}]
let loose = { actions: [] }; // actions when there is no plan
let startedAt = 0;
let timer = null;

function icon(name) {
  const i = document.createElement('i');
  i.className = `icon i-${name}`;
  return i;
}

function el(tag, className, ...children) {
  const e = document.createElement(tag);
  if (className) e.className = className;
  e.append(...children.filter((c) => c !== null && c !== undefined));
  return e;
}

// --- status + clock -----------------------------------------------------------

function setStatus(state) {
  $('status').dataset.state = state;
  $('status-text').textContent = state;
  room.setState(state);
}

function clock(on) {
  clearInterval(timer);
  if (on) {
    startedAt = performance.now();
    const tick = () => {
      const s = Math.floor((performance.now() - startedAt) / 1000);
      $('timer').textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
    };
    tick();
    timer = setInterval(tick, 500);
  }
}

// --- the timeline -------------------------------------------------------------

// 'click u112 Button "Rectangle"' -> 'clicked Rectangle': the log reads like a person wrote it.
// The exact label stays in the tooltip.
function humanize(a) {
  const quoted = a.label.match(/"([^"]+)"/)?.[1];
  const typed = a.label.match(/type '([^']*)'|type "([^"]*)"|\s(\S+\.\w{2,4})$/);
  switch (a.kind) {
    case 'click': return quoted ? `clicked ${quoted}` : 'clicked on the screen';
    case 'drag': return 'drew on the canvas';
    case 'type': return `typed ${typed?.[1] ?? typed?.[2] ?? typed?.[3] ?? 'text'}`;
    case 'key': return `pressed ${a.label.replace(/^key\s+/, '').split('+').map((k) => k[0].toUpperCase() + k.slice(1)).join('+')}`;
    case 'locate': return `looked for the ${quoted ?? 'target'}`;
    case 'scroll': return 'scrolled';
    default: return a.label;
  }
}

function actionRow(a) {
  const row = el('div', 'action', el('span', 'dot'), el('span', 'label', humanize(a)), a.channel ? el('span', 'via', a.channel) : null);
  if (a.channel) row.dataset.channel = a.channel;
  row.title = a.note ? `${a.label}\n${a.note}` : a.label;
  return row;
}

// Rows are made once and then only updated: re-creating them would restart their entrance
// animation on every event, and the list would flicker.
function buildTimeline() {
  const list = $('timeline');
  list.replaceChildren();
  plan.forEach((p, i) => {
    p.node = el('span', 'node');
    p.actionsEl = el('div', 'actions');
    p.li = el('li', 'step', p.node, el('span', 'text', p.text), p.actionsEl);
    p.index = i;
    list.append(p.li);
  });
  if (!plan.length) {
    loose.node = el('span', 'node');
    loose.actionsEl = el('div', 'actions');
    loose.li = el('li', 'step', loose.node, el('span', 'text', 'Working'), loose.actionsEl);
  }
}

function addAction(a) {
  const owner = currentStep() ?? loose;
  owner.actions.push(a);
  if (owner === loose && !loose.li.isConnected) $('timeline').append(loose.li);
  owner.actionsEl.append(actionRow(a));
  while (owner.actionsEl.children.length > 4) owner.actionsEl.firstChild.remove();
}

function setReceipt(p, receipt) {
  p.receipt = receipt;
  p.li.querySelector('.receipt')?.remove();
  const ok = receipt.verdict === 'verified';
  p.li.append(el('div', `receipt${ok ? '' : ' bad'}`, `${ok ? 'verified' : 'not verified'}: ${receipt.summary}`));
}

function render() {
  const now = plan.findIndex((p) => !p.done);
  plan.forEach((p, i) => {
    const state = p.done ? 'done' : i === now && running ? 'now' : '';
    if (p.li.dataset.state === state) return;
    p.li.dataset.state = state;
    p.li.className = `step ${state}`;
    p.node.replaceChildren(p.done ? '✓' : String(i + 1));
  });
  if (loose.li) loose.li.className = `step ${running ? 'now' : 'done'}`;
  const n = plan.reduce((sum, p) => sum + p.actions.length, 0) + loose.actions.length;
  const done = plan.filter((p) => p.done).length;
  const parts = [appLabel, plan.length ? `${done} of ${plan.length} done` : null, n ? `${n} action${n === 1 ? '' : 's'}` : null];
  $('meta').textContent = parts.filter(Boolean).join(' · ');
}

function currentStep() {
  return plan.find((p) => !p.done) ?? plan[plan.length - 1];
}

// --- reasoning ----------------------------------------------------------------

let typingJob = null;
function addThought(text) {
  const box = $('thoughts');
  box.querySelector('.placeholder')?.remove();
  box.querySelectorAll('p.current').forEach((p) => p.classList.remove('current'));
  const p = el('p', 'current');
  box.append(p);
  while (box.children.length > 5) box.firstChild.remove();
  cancelAnimationFrame(typingJob);
  let shown = 0;
  let last = performance.now();
  const step = (now) => {
    shown = Math.min(text.length, shown + Math.max(1, Math.round((now - last) * 0.09)));
    last = now;
    p.textContent = text.slice(0, shown);
    box.scrollTop = box.scrollHeight;
    if (shown < text.length) typingJob = requestAnimationFrame(step);
  };
  typingJob = requestAnimationFrame(step);
  room.think();
}

// --- the screen panel -----------------------------------------------------------

function showScreen(src) {
  const img = $('shot-img');
  img.onload = () => { img.classList.add('on'); $('noshot').hidden = true; };
  img.src = src;
  room.setScreen({ image: src });
}

function pointAt(x, y, channel) {
  const m = $('marker');
  m.style.left = `${x * 100}%`;
  m.style.top = `${y * 100}%`;
  m.style.setProperty('--c', CHANNEL_COLOR[channel] ?? 'var(--work)');
  m.classList.add('on');
  m.classList.remove('ping');
  void m.offsetWidth; // restart the ping animation
  m.classList.add('ping');
}

// --- results ---------------------------------------------------------------------

function showResult(state, text, meta) {
  const box = $('result');
  box.hidden = false;
  box.dataset.state = state;
  box.replaceChildren(
    el('span', 'seal', icon(state === 'done' ? 'check' : 'x')),
    el('div', '', el('div', 'what', text), meta ? el('div', 'meta', meta) : null),
  );
}

function setRunning(on) {
  running = on;
  $('go').disabled = on;
  $('goal').disabled = on;
  $('stop').disabled = !on;
  for (const b of $('targets').children) b.disabled = on;
  render();
}

// --- events from the agent ----------------------------------------------------------

window.agent.onEvent((msg) => {
  const p = msg.params ?? {};
  switch (msg.method) {
    case 'planning':
      setStatus('planning');
      $('goal-title').textContent = p.goal;
      $('goal-title').classList.remove('empty');
      room.setScreen({ app: '', goal: p.goal, lines: [], answer: '' });
      room.think();
      break;
    case 'plan': {
      plan = p.steps.map((text) => ({ text, done: false, actions: [], receipt: null }));
      buildTimeline();
      if (p.goal) $('goal-title').textContent = p.goal;
      const t = TARGETS[p.app];
      appLabel = t?.label ?? p.app ?? '';
      room.setScreen({ app: t?.label ?? p.app, goal: p.goal });
      render();
      break;
    }
    case 'started':
      setStatus('working');
      clock(true);
      room.setScreen({ app: TARGETS[p.app]?.label ?? p.app, ...(p.goal ? { goal: p.goal } : {}), lines: [], answer: '' });
      room.act('start');
      break;
    case 'thought':
      addThought(p.text);
      break;
    case 'screen':
      showScreen(p.src);
      break;
    case 'step': {
      if (p.kind === 'done') return; // the ending is told by `finished`
      addAction(p);
      $('last').textContent = `${humanize(p)} · ${p.channel ?? ''}`;
      const point = p.x !== undefined ? { x: p.x, y: p.y, channel: p.channel } : null;
      if (point) pointAt(p.x, p.y, p.channel);
      room.act(p.kind, point);
      render();
      break;
    }
    case 'receipt':
      if (plan[p.step - 1]) setReceipt(plan[p.step - 1], p);
      render();
      break;
    case 'check':
      if (plan[p.step - 1]) plan[p.step - 1].done = true;
      room.act('check');
      render();
      break;
    case 'finished': {
      setRunning(false);
      clock(false);
      const state = ['done', 'stopped', 'error'].includes(p.status) ? p.status : 'stopped';
      setStatus(state);
      const answer = (p.answer || '').replace(/^STOPPED:\s*/, '');
      room.setScreen({ answer });
      const verified = plan.filter((s) => s.receipt?.verdict === 'verified').length;
      const meta = [p.seconds !== undefined ? `${p.seconds}s` : null, p.actions !== undefined ? `${p.actions} actions` : null,
        plan.length ? `${verified}/${plan.length} checkpoints verified` : null].filter(Boolean).join(' · ');
      showResult(state, answer, meta);
      break;
    }
    case 'approval':
      askPerson(p.id, p.action);
      break;
    case 'approval-done':
      if ($('ask').dataset.id === p.id) $('ask').hidden = true;
      break;
    case 'sidecar-exit':
      setRunning(false);
      clock(false);
      setStatus('error');
      showResult('error', `The agent process stopped (${p.error ?? `exit code ${p.code}`}). Restart the app.`);
      break;
  }
});

async function run(goal) {
  if (!goal.trim() || running) return;
  $('result').hidden = true;
  plan = [];
  loose = { actions: [] };
  appLabel = '';
  buildTimeline();
  $('thoughts').replaceChildren(el('p', 'placeholder', 'Thinking…'));
  setRunning(true);
  const reply = await window.agent.request('run', target ? { goal, app: target } : { goal });
  if (reply.error) {
    setRunning(false);
    setStatus('error');
    showResult('error', reply.error.message);
  }
}

$('command').addEventListener('submit', (e) => {
  e.preventDefault();
  run($('goal').value);
});
$('stop').addEventListener('click', () => window.agent.request('stop'));

// An irreversible action waits for the person: the answer goes back through the sidecar to the
// MCP server, which is blocked on it (session.file_approver).
function askPerson(id, action) {
  $('ask').dataset.id = id;
  $('ask-what').textContent = action;
  $('ask').hidden = false;
  $('allow').focus();
}
async function answer(ok) {
  const id = $('ask').dataset.id;
  $('ask').hidden = true;
  if (id) await window.agent.request('approve', { id, ok });
}
$('allow').addEventListener('click', () => answer(true));
$('deny').addEventListener('click', () => answer(false));

function pickTarget(name) {
  target = name;
  for (const b of $('targets').children) b.setAttribute('aria-checked', String(b.dataset.target === name));
}

(async () => {
  const ping = await window.agent.request('ping');
  const brain = ping.result?.brain ?? '';
  $('brain-name').textContent = DEMO ? 'demo · scripted, no mouse' : brain === 'claude' ? 'Claude · via MCP · moves your mouse' : `${brain} · moves your mouse`;
  $('goal').placeholder = EXAMPLES[0];
  let n = 0;
  setInterval(() => { if (!$('goal').value) $('goal').placeholder = EXAMPLES[++n % EXAMPLES.length]; }, 4000);
  const reply = await window.agent.request('apps');
  for (const name of ['', ...(reply.result?.apps ?? [])]) {
    const t = TARGETS[name] ?? { label: name, icon: 'box' };
    const b = el('button', 'target', icon(t.icon), el('span', '', t.label));
    b.type = 'button';
    b.dataset.target = name;
    b.setAttribute('role', 'radio');
    b.addEventListener('click', () => pickTarget(name));
    $('targets').append(b);
  }
  pickTarget('');
  if (reply.error) showResult('error', reply.error.message);
  render();
  $('goal').focus();
  const autorun = params.get('autorun');
  if (autorun) {
    $('goal').value = autorun;
    run(autorun);
  }
})();
