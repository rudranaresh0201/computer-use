// The 3D room: a robot at a console with a monitor. The page tells it what the
// agent is doing (room.act('click'), room.setState('done'), room.setScreen(...))
// and the room acts it out. Nothing here knows about JSON-RPC.
//
// Robot: RobotExpressive by Tomás Laulhé (CC0), from the three.js examples.
// Its clips: Idle Walking Running Dance Death Sitting Standing (loops) and
// Jump Yes No Wave Punch ThumbsUp (one-shots).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLORS = { idle: 0x8a919c, planning: 0x4cc9f0, working: 0x4cc9f0, done: 0x4ade80, stopped: 0xfbbf24, error: 0xf87171 };
const CSS = { idle: '#8a919c', planning: '#4cc9f0', working: '#4cc9f0', done: '#4ade80', stopped: '#fbbf24', error: '#f87171' };
const EMOTE = { click: 'Punch', type: 'Punch', key: 'Punch', scroll: 'Yes', noop: 'No', start: 'Wave', check: 'Yes' };
const FACE_SCREEN = -2.25; // robot's heading toward the monitor
const FACE_CAMERA = 0.45; // ...and toward you

export class Room {
  constructor(canvas) {
    this.state = 'idle';
    this.screen = { app: '', goal: '', status: 'idle', lines: [], answer: '' };
    this.clock = new THREE.Clock();
    this.targetHeading = FACE_CAMERA;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d10);
    scene.fog = new THREE.Fog(0x0b0d10, 7, 16);
    this.scene = scene;

    this.camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);
    this.camera.position.set(2.7, 2.1, 4.2);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(-0.1, 1.15, -0.4);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    this.controls.minDistance = 3;
    this.controls.maxDistance = 8;
    this.controls.maxPolarAngle = Math.PI * 0.49;

    scene.add(new THREE.HemisphereLight(0x9fb4c8, 0x0b0d10, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(3, 6, 4);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    Object.assign(key.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4 });
    scene.add(key);
    this.glow = new THREE.PointLight(COLORS.idle, 6, 4, 1.6);
    this.glow.position.set(-0.5, 1.5, -0.3);
    scene.add(this.glow);

    this.#buildFloor();
    this.#buildConsole();
    this.#buildDust();
    this.#loadRobot();

    window.addEventListener('resize', () => this.#resize());
    this.#resize();
    renderer.setAnimationLoop(() => this.#frame());
  }

  // --- what the page calls -------------------------------------------------

  setState(state) {
    this.state = state;
    this.screen.status = state;
    const color = COLORS[state] ?? COLORS.idle;
    this.ring.material.color.setHex(color);
    this.glow.color.setHex(color);
    this.targetHeading = state === 'working' ? FACE_SCREEN : FACE_CAMERA;
    if (state === 'done') this.#emote('ThumbsUp', 'Dance', 2.6);
    else if (state === 'stopped') this.#emote('No');
    else if (state === 'error') this.#emote('Death', 'Idle', 2.4);
    this.#drawScreen();
  }

  act(kind) {
    if (EMOTE[kind]) this.#emote(EMOTE[kind]);
    this.ringPulse = 1;
  }

  setScreen(patch) {
    Object.assign(this.screen, patch);
    this.#drawScreen();
  }

  // --- building the room ---------------------------------------------------

  #buildFloor() {
    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(12, 64),
      new THREE.MeshStandardMaterial({ color: 0x111418, roughness: 0.95 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);
    const grid = new THREE.GridHelper(24, 48, 0x1c2128, 0x161a20);
    grid.position.y = 0.002;
    this.scene.add(grid);

    this.ring = new THREE.Mesh(
      new THREE.RingGeometry(0.55, 0.62, 64),
      new THREE.MeshBasicMaterial({ color: COLORS.idle, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending }),
    );
    this.ring.rotation.x = -Math.PI / 2;
    this.ring.position.set(0.75, 0.01, 0.15);
    this.ringPulse = 0;
    this.scene.add(this.ring);
  }

  #buildConsole() {
    const metal = new THREE.MeshStandardMaterial({ color: 0x1a1f26, roughness: 0.5, metalness: 0.6 });
    const group = new THREE.Group();
    group.position.set(-0.5, 0, -0.75);
    group.rotation.y = 0.25;

    const top = new THREE.Mesh(new THREE.BoxGeometry(2.0, 0.06, 0.8), metal);
    top.position.y = 0.9;
    top.castShadow = top.receiveShadow = true;
    group.add(top);
    for (const x of [-0.9, 0.9]) {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.9, 0.7), metal);
      leg.position.set(x, 0.45, 0);
      leg.castShadow = true;
      group.add(leg);
    }

    const bezel = new THREE.Mesh(new THREE.BoxGeometry(1.72, 1.06, 0.05), metal);
    bezel.position.set(0, 1.55, -0.2);
    bezel.castShadow = true;
    group.add(bezel);
    const stand = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.12, 0.08), metal);
    stand.position.set(0, 0.99, -0.2);
    group.add(stand);

    this.screenCanvas = document.createElement('canvas');
    this.screenCanvas.width = 1024;
    this.screenCanvas.height = 620;
    this.screenTexture = new THREE.CanvasTexture(this.screenCanvas);
    this.screenTexture.colorSpace = THREE.SRGBColorSpace;
    this.screenTexture.anisotropy = 8;
    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(1.62, 0.98),
      new THREE.MeshBasicMaterial({ map: this.screenTexture, toneMapped: false }),
    );
    screen.position.set(0, 1.55, -0.172);
    group.add(screen);

    this.scene.add(group);
    this.#drawScreen();
  }

  #buildDust() {
    const n = 260;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 10;
      positions[i * 3 + 1] = Math.random() * 4;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 10;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.dust = new THREE.Points(geometry, new THREE.PointsMaterial({
      color: 0x4cc9f0, size: 0.018, transparent: true, opacity: 0.45, depthWrite: false,
    }));
    this.scene.add(this.dust);
  }

  #loadRobot() {
    new GLTFLoader().load('../assets/RobotExpressive.glb', (gltf) => {
      const robot = gltf.scene;
      const box = new THREE.Box3().setFromObject(robot);
      robot.scale.setScalar(1.7 / (box.max.y - box.min.y));
      robot.position.set(0.75, 0, 0.15);
      robot.rotation.y = FACE_CAMERA;
      robot.traverse((o) => { if (o.isMesh) o.castShadow = true; });
      this.scene.add(robot);
      this.robot = robot;

      this.mixer = new THREE.AnimationMixer(robot);
      this.actions = {};
      for (const clip of gltf.animations) {
        const action = this.mixer.clipAction(clip);
        if (['Jump', 'Yes', 'No', 'Wave', 'Punch', 'ThumbsUp', 'Death'].includes(clip.name)) {
          action.clampWhenFinished = true;
          action.loop = THREE.LoopOnce;
        }
        this.actions[clip.name] = action;
      }
      this.base = this.actions.Idle;
      this.current = this.base;
      this.base.play();
      this.mixer.addEventListener('finished', (e) => this.#backToBase(e.action));
      window.__robotClips = Object.keys(this.actions); // for the self-check
    });
  }

  // --- animation -------------------------------------------------------------

  #fadeTo(action, seconds = 0.25) {
    if (!action) return;
    if (action === this.current) {
      if (action.loop === THREE.LoopOnce) action.reset().play(); // a second click punches again
      return;
    }
    if (this.current) this.current.fadeOut(seconds);
    action.reset().setEffectiveWeight(1).fadeIn(seconds).play();
    this.current = action;
  }

  // a one-shot, then settle into `then` (a loop) -- the robot stays in Dance after a success
  #emote(name, then = 'Idle', hold = 0) {
    if (!this.actions) return;
    this.nextBase = this.actions[then] ?? this.actions.Idle;
    this.hold = hold;
    this.#fadeTo(this.actions[name], 0.15);
  }

  #backToBase(finished) {
    const hold = this.hold;
    this.hold = 0;
    setTimeout(() => {
      if (this.current !== finished) return; // a newer emote is already playing
      this.base = this.nextBase ?? this.actions.Idle;
      this.#fadeTo(this.base, 0.35);
    }, hold * 1000);
  }

  #frame() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    this.mixer?.update(dt);
    if (this.robot) {
      const diff = Math.atan2(Math.sin(this.targetHeading - this.robot.rotation.y), Math.cos(this.targetHeading - this.robot.rotation.y));
      this.robot.rotation.y += diff * Math.min(1, dt * 4);
    }
    this.ringPulse = Math.max(0, this.ringPulse - dt * 1.8);
    const breathing = this.state === 'working' ? 0.15 * Math.sin(t * 4) : 0.05 * Math.sin(t * 1.5);
    this.ring.material.opacity = 0.45 + breathing + this.ringPulse * 0.5;
    this.ring.scale.setScalar(1 + this.ringPulse * 0.25);
    this.glow.intensity = (this.state === 'working' ? 7 : 4) + this.ringPulse * 6;
    const p = this.dust.geometry.attributes.position;
    for (let i = 0; i < p.count; i++) {
      let y = p.getY(i) + dt * 0.06;
      if (y > 4) y = 0;
      p.setY(i, y);
    }
    p.needsUpdate = true;
    if (Math.floor(t * 2) !== this.blink) { this.blink = Math.floor(t * 2); this.#drawScreen(); }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  #resize() {
    const { innerWidth: w, innerHeight: h } = window;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // --- the monitor: a canvas drawn as a texture ------------------------------

  #drawScreen() {
    const c = this.screenCanvas?.getContext('2d');
    if (!c) return;
    const { width: W, height: H } = this.screenCanvas;
    const s = this.screen;
    const accent = CSS[s.status] ?? CSS.idle;
    c.fillStyle = '#07090c';
    c.fillRect(0, 0, W, H);
    c.fillStyle = '#10141a';
    c.fillRect(0, 0, W, 64);
    c.fillStyle = accent;
    c.beginPath();
    c.arc(36, 32, 9, 0, Math.PI * 2);
    c.fill();
    c.font = '600 28px Consolas, monospace';
    c.fillStyle = '#e6e8eb';
    c.fillText(s.app ? `agent @ ${s.app}` : 'agent', 60, 42);
    c.fillStyle = accent;
    c.textAlign = 'right';
    c.fillText(s.status, W - 30, 42);
    c.textAlign = 'left';

    c.font = '24px Consolas, monospace';
    c.fillStyle = '#8a919c';
    c.fillText(s.goal ? `> ${s.goal}`.slice(0, 60) : '> waiting for a command', 30, 112);

    const lines = s.lines.slice(-11);
    lines.forEach((line, i) => {
      const last = i === lines.length - 1 && s.status === 'working';
      c.fillStyle = line.kind === 'noop' ? CSS.stopped : last ? CSS.working : '#c5cad1';
      c.fillText(`${String(line.n).padStart(2, ' ')}  ${line.text}`.slice(0, 64), 30, 160 + i * 34);
    });
    if (s.status === 'working' && this.blink % 2 === 0) {
      c.fillStyle = CSS.working;
      c.fillRect(30, 160 + lines.length * 34 - 22, 14, 26);
    }
    if (s.answer) {
      c.fillStyle = '#10141a';
      c.fillRect(0, H - 92, W, 92);
      c.font = '600 34px Consolas, monospace';
      c.fillStyle = accent;
      c.fillText(s.answer.slice(0, 48), 30, H - 34);
    }
    this.screenTexture.needsUpdate = true;
  }
}
