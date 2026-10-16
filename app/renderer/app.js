// The page: turns the agent's notifications into what you see.
//
//   planning -> status "planning", the robot thinks
//   plan     -> the checklist on the left (the planner's checkpoints)
//   started  -> robot waves, turns to the monitor, status "working"
//   step     -> a line on the monitor; the robot acts out step.kind
//   check    -> a checkpoint ticks; the robot nods
//   finished -> answer card; thumbs-up + dance, or "no", or a fall
//
// Unticked checkpoints stay unticked at the end: the list shows what the
// agent claimed, not what we hoped.
//
// window.agent comes from preload.js: request(method, params) and onEvent(cb).

import { Room } from './scene.js';

const $ = (id) => document.getElementById(id);
const room = new Room($('scene'));
const params = new URLSearchParams(location.search);
let running = false;
let plan = []; // [{text, done}]
let actions = []; // step params

$('mode').textContent = params.get('mode') === 'demo' ? 'demo mode · scripted, no mouse' : 'live · the agent moves your mouse';

function setStatus(state) {
  $('status').dataset.state = state === 'planning' ? 'working' : state;
  $('status').textContent = state;
  room.setState(state);
}

function li(className, text, note) {
  const item = document.createElement('li');
  item.className = className;
  const span = document.createElement('span');
  span.textContent = text;
  item.append(span);
  if (note) {
    const small = document.createElement('span');
    small.className = 'note';
    small.textContent = note;
    item.append(small);
  }
  return item;
}

function render() {
  const list = $('steps');
  list.replaceChildren();
  if (plan.length) {
    const next = plan.findIndex((p) => !p.done);
    plan.forEach((p, i) => list.append(li(p.done ? 'ok' : i === next && running ? 'now' : '', p.text)));
  } else if (actions.length) {
    actions.forEach((a, i) => list.append(li(a.kind === 'noop' ? 'noop' : i === actions.length - 1 && running ? 'now' : 'ok', a.label, a.note)));
  } else {
    list.append(li('empty', running ? 'Planning…' : 'Nothing yet. Tell it what to do below.'));
  }
  const last = actions[actions.length - 1];
  $('last').textContent = last ? `${actions.length} action${actions.length === 1 ? '' : 's'} · last: ${last.label}` : '';
}

function setRunning(on) {
  running = on;
  for (const id of ['go', 'goal', 'app']) $(id).disabled = on;
  $('stop').disabled = !on;
  render();
}

function showAnswer(state, text) {
  const box = $('answer');
  box.hidden = false;
  box.dataset.state = state;
  box.replaceChildren();
  const label = document.createElement('span');
  label.className = 'label';
  label.textContent = { done: 'answer', stopped: 'stopped', error: 'error' }[state] ?? state;
  box.append(label, document.createTextNode(text));
}

window.agent.onEvent((msg) => {
  const p = msg.params ?? {};
  switch (msg.method) {
    case 'planning':
      setStatus('planning');
      room.setScreen({ app: '', goal: p.goal, lines: [], answer: '' });
      break;
    case 'plan':
      plan = p.steps.map((text) => ({ text, done: false }));
      room.setScreen({ app: p.app, goal: p.goal });
      render();
      break;
    case 'started':
      room.setScreen({ app: p.app, goal: p.goal, lines: [], answer: '' });
      setStatus('working');
      room.act('start');
      break;
    case 'step':
      if (p.kind === 'done') return; // the ending is told by `finished`
      actions.push(p);
      room.setScreen({ lines: actions.map((a) => ({ n: a.n, kind: a.kind, text: a.label })) });
      room.act(p.kind);
      render();
      break;
    case 'check':
      if (plan[p.step - 1]) plan[p.step - 1].done = true;
      room.act('check');
      render();
      break;
    case 'finished': {
      setRunning(false);
      const state = ['done', 'stopped', 'error'].includes(p.status) ? p.status : 'stopped';
      setStatus(state);
      const answer = (p.answer || '').replace(/^STOPPED:\s*/, '');
      room.setScreen({ answer });
      const extra = p.seconds !== undefined ? ` · ${p.actions} actions, ${p.seconds}s` : '';
      showAnswer(state, answer + extra);
      break;
    }
    case 'sidecar-exit':
      setRunning(false);
      setStatus('error');
      showAnswer('error', `The agent process stopped (${p.error ?? `exit code ${p.code}`}). Restart the app.`);
      break;
  }
});

async function run(goal) {
  if (!goal.trim() || running) return;
  $('answer').hidden = true;
  plan = [];
  actions = [];
  setRunning(true);
  const app = $('app').value;
  const reply = await window.agent.request('run', app ? { goal, app } : { goal });
  if (reply.error) {
    setRunning(false);
    setStatus('error');
    showAnswer('error', reply.error.message);
  }
}

$('command').addEventListener('submit', (e) => {
  e.preventDefault();
  run($('goal').value);
});
$('stop').addEventListener('click', () => window.agent.request('stop'));

(async () => {
  $('app').append(new Option('auto · it picks the app', ''));
  const reply = await window.agent.request('apps');
  for (const name of reply.result?.apps ?? []) $('app').append(new Option(name, name));
  if (reply.error) showAnswer('error', reply.error.message);
  render();
  $('goal').focus();
  const autorun = params.get('autorun');
  if (autorun) {
    $('goal').value = autorun;
    run(autorun);
  }
})();
