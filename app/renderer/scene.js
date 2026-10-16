// The 3D room: a robot that works at a real desk. The page tells it what the agent is doing
// (room.setState, room.act, room.think, room.setScreen) and the room acts it out.
// Nothing here knows about JSON-RPC.
//
// Light and background: Poly Haven's "hotel_room" HDRI (CC0), projected onto the ground so
// the props stand on the photographed floor. Props: Poly Haven models (CC0), real-world metres.
// Robot: RobotExpressive by Tomás Laulhé (CC0). Clips: Idle Walking Running Dance Death Sitting
// Standing (loops), Jump Yes No Wave Punch ThumbsUp (one-shots).
//
// Choreography: idle = standing beside the desk, facing you. working = walks to the chair, sits,
// looks at the monitor (its eyes follow where the agent points), types on type/key, moves the
// mouse arm on click/drag, nods (head only) on a verified checkpoint. done = stands up, thumbs up.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GroundedSkybox } from 'three/addons/objects/GroundedSkybox.js';
import { RoundedBoxGeometry } from 'three/addons/geometries/RoundedBoxGeometry.js';
import { RectAreaLightUniformsLib } from 'three/addons/lights/RectAreaLightUniformsLib.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { GTAOPass } from 'three/addons/postprocessing/GTAOPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { BokehPass } from 'three/addons/postprocessing/BokehPass.js';
import { FilmPass } from 'three/addons/postprocessing/FilmPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const PH = '../assets/ph';
export const STATE_COLOR = {
  idle: '#9aa3ad', planning: '#7dd3fc', working: '#7dd3fc', done: '#86efac', stopped: '#fcd34d', error: '#fca5a5',
};
export const CHANNEL_COLOR = { tree: '#60a5fa', eyes: '#fb923c', vision: '#f472b6', api: '#a78bfa' };

const DESK_TOP = 0.79;
const ROBOT_HEIGHT = 0.9;
const SKY_TURN = Number(new URLSearchParams(location.search).get('sky') ?? 3.14); // which part of the photo is behind the desk
const SPOT = {
  stand: new THREE.Vector3(1.3, 0, 0.35), // beside the desk, facing you
  chairFront: new THREE.Vector3(0.05, 0, 1.25),
  seat: new THREE.Vector3(0.05, 0.07, 0.86),
};
const HEADING = { you: Math.atan2(3.4 - 1.3, 3.7 - 0.35), desk: Math.PI };
const SHOTS = {
  idle: { pos: [3.4, 1.65, 3.7], target: [0.35, 0.5, 0.05], focus: 3.9 },
  working: { pos: [2.3, 1.5, 2.25], target: [-0.25, 0.8, 0.3], focus: 2.6 },
  done: { pos: [3.3, 1.55, 3.1], target: [0.75, 0.38, 0.2], focus: 2.8 },
};

const ease = (t) => t * t * (3 - 2 * t);
const damp = (current, target, lambda, dt) => current + (target - current) * (1 - Math.exp(-lambda * dt));

export class Room {
  constructor(canvas) {
    this.state = 'idle';
    this.screen = { app: '', goal: '', status: 'idle', lines: [], answer: '', image: null, point: null };
    this.clock = new THREE.Clock();
    this.pointer = { x: 0.5, y: 0.5 }; // where on the monitor the robot looks (0..1)
    this.typing = 0;
    this.mousing = 0;
    this.thinking = 0;
    this.manualUntil = 0;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.VSMShadowMap;
    renderer.toneMapping = THREE.AgXToneMapping;
    renderer.toneMappingExposure = 0.85;
    this.renderer = renderer;

    const scene = new THREE.Scene();
    this.scene = scene;
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.05, 80);
    this.camera.position.set(...SHOTS.idle.pos);
    this.lookAt = new THREE.Vector3(...SHOTS.idle.target);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.copy(this.lookAt);
    Object.assign(this.controls, {
      enableDamping: true, enablePan: false, minDistance: 1.6, maxDistance: 6, maxPolarAngle: Math.PI * 0.47,
    });
    this.controls.addEventListener('start', () => { this.manualUntil = this.clock.elapsedTime + 8; });

    RectAreaLightUniformsLib.init();
    this.#buildLights();
    this.#buildMonitorAndKeyboard();
    this.#loadEnvironment();
    this.#loadProps();
    this.#loadRobot();
    this.#buildComposer();

    window.addEventListener('resize', () => this.#resize());
    this.#resize();
    renderer.setAnimationLoop(() => this.#frame());
  }

  // --- what the page calls -------------------------------------------------

  setState(state) {
    const was = this.state;
    this.state = state;
    this.screen.status = state;
    const color = new THREE.Color(STATE_COLOR[state] ?? STATE_COLOR.idle);
    this.screenLight.color.copy(color).lerp(new THREE.Color('#dfe9ff'), 0.6);
    if (state === 'working' && was !== 'working') this.#goSit();
    if (['done', 'stopped', 'error'].includes(state)) this.#goStand(state);
    if (state === 'idle') this.#goStand('idle');
    this.#drawScreen();
  }

  act(kind, point) {
    if (point) this.pointer = { x: point.x, y: point.y };
    if (kind === 'type' || kind === 'key') this.typing = 1;
    else if (kind === 'click' || kind === 'drag' || kind === 'scroll') this.mousing = 1;
    else if (kind === 'locate') this.thinking = Math.max(this.thinking, 0.6);
    else if (kind === 'check') this.#nod();
    else if (kind === 'start') this.#emote('Wave');
    if (point) this.screen.point = { ...point, t: this.clock.elapsedTime };
    this.#drawScreen();
  }

  think() {
    this.thinking = 1;
  }

  setScreen(patch) {
    if (patch.image !== undefined && patch.image !== this.screen.imageSrc) {
      this.screen.imageSrc = patch.image;
      if (patch.image) {
        const img = new Image();
        img.onload = () => { this.screen.image = img; this.#drawScreen(); };
        img.src = patch.image;
      } else this.screen.image = null;
    }
    const { image, ...rest } = patch;
    Object.assign(this.screen, rest);
    this.#drawScreen();
  }

  // --- the world -------------------------------------------------------------

  #loadEnvironment() {
    new RGBELoader().load(`${PH}/hotel_room.hdr`, (hdr) => {
      hdr.mapping = THREE.EquirectangularReflectionMapping;
      this.scene.environment = hdr;
      this.scene.environmentIntensity = 0.42;
      // the photo becomes the room: projected onto a ground at the camera's capture height
      const sky = new GroundedSkybox(hdr, 1.55, 14);
      sky.position.y = 1.55 - 0.001;
      sky.rotation.y = SKY_TURN;
      this.scene.environmentRotation.y = SKY_TURN;
      sky.material.color = new THREE.Color(0.42, 0.4, 0.38); // evening: the room darker than the desk
      this.scene.add(sky);
      this.ready = true;
    });
  }

  #buildLights() {
    // warm desk lamp: the key light, the one that casts the robot's shadow on the desk
    const lamp = new THREE.SpotLight('#ffd9a8', 9, 3.2, Math.PI / 4.2, 0.65, 1.6);
    lamp.position.set(-0.78, 1.42, -0.2);
    lamp.target.position.set(0.05, DESK_TOP, 0.25);
    lamp.castShadow = true;
    lamp.shadow.mapSize.set(2048, 2048);
    lamp.shadow.radius = 6;
    lamp.shadow.bias = -0.0004;
    this.scene.add(lamp, lamp.target);
    this.lampBulb = lamp;

    // cool window fill from the side, soft, for shape and the floor shadow
    const fill = new THREE.DirectionalLight('#bcd2ff', 0.55);
    fill.position.set(4, 5, 2.5);
    fill.castShadow = true;
    fill.shadow.mapSize.set(2048, 2048);
    fill.shadow.radius = 12;
    fill.shadow.blurSamples = 16;
    Object.assign(fill.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 14 });
    this.scene.add(fill);

    // the monitor lights the robot's face in the state colour
    this.screenLight = new THREE.RectAreaLight('#dfe9ff', 5, 0.6, 0.34);
    this.screenLight.position.set(0, DESK_TOP + 0.33, -0.2);
    this.screenLight.lookAt(0, DESK_TOP + 0.33, 2);
    this.scene.add(this.screenLight);
  }

  #buildMonitorAndKeyboard() {
    const alu = new THREE.MeshStandardMaterial({ color: '#2a2d31', metalness: 0.75, roughness: 0.32 });
    const black = new THREE.MeshStandardMaterial({ color: '#0a0b0d', metalness: 0.2, roughness: 0.55 });
    const group = new THREE.Group();
    group.position.set(0, DESK_TOP, -0.24);

    const panel = new THREE.Mesh(new RoundedBoxGeometry(0.64, 0.385, 0.022, 4, 0.008), black);
    panel.position.y = 0.33;
    const neck = new THREE.Mesh(new RoundedBoxGeometry(0.05, 0.2, 0.018, 3, 0.006), alu);
    neck.position.set(0, 0.12, -0.035);
    neck.rotation.x = -0.12;
    const base = new THREE.Mesh(new RoundedBoxGeometry(0.22, 0.012, 0.16, 3, 0.005), alu);
    base.position.set(0, 0.006, -0.02);
    for (const m of [panel, neck, base]) { m.castShadow = true; m.receiveShadow = true; group.add(m); }

    this.screenCanvas = document.createElement('canvas');
    this.screenCanvas.width = 1280;
    this.screenCanvas.height = 720;
    this.screenTexture = new THREE.CanvasTexture(this.screenCanvas);
    this.screenTexture.colorSpace = THREE.SRGBColorSpace;
    this.screenTexture.anisotropy = 8;
    const glass = new THREE.Mesh(
      new THREE.PlaneGeometry(0.616, 0.3465),
      new THREE.MeshBasicMaterial({ map: this.screenTexture, toneMapped: false, color: new THREE.Color(1.15, 1.15, 1.15) }),
    );
    glass.position.set(0, 0.333, 0.0115);
    group.add(glass);
    this.scene.add(group);

    // keyboard: a slab and instanced keys
    const kb = new THREE.Group();
    kb.position.set(0.02, DESK_TOP, 0.2);
    const slab = new THREE.Mesh(new RoundedBoxGeometry(0.38, 0.012, 0.125, 3, 0.005), alu);
    slab.position.y = 0.006;
    slab.castShadow = slab.receiveShadow = true;
    kb.add(slab);
    const keys = new THREE.InstancedMesh(new RoundedBoxGeometry(0.017, 0.006, 0.017, 2, 0.003), black, 15 * 5);
    const m = new THREE.Matrix4();
    let i = 0;
    for (let r = 0; r < 5; r++) for (let c = 0; c < 15; c++) {
      m.makeTranslation(-0.165 + c * 0.0236, 0.015, -0.045 + r * 0.0225);
      keys.setMatrixAt(i++, m);
    }
    keys.castShadow = true;
    kb.add(keys);
    const mouse = new THREE.Mesh(new THREE.CapsuleGeometry(0.022, 0.035, 6, 16), black);
    mouse.rotation.x = Math.PI / 2;
    mouse.scale.set(1, 1, 0.45);
    mouse.position.set(0.3, 0.012, 0.03);
    mouse.castShadow = true;
    kb.add(mouse);
    this.scene.add(kb);
    this.drawScreenSoon = true;
  }

  #loadProps() {
    const loader = new GLTFLoader();
    const place = (id, position, rotationY = 0, scale = 1) => loader.load(`${PH}/${id}/${id}.gltf`, (gltf) => {
      const o = gltf.scene;
      o.position.copy(position);
      o.rotation.y = rotationY;
      o.scale.setScalar(scale);
      o.traverse((c) => { if (c.isMesh) { c.castShadow = true; c.receiveShadow = true; } });
      this.scene.add(o);
      this.props ??= {};
      this.props[id] = o;
    });
    place('metal_office_desk', new THREE.Vector3(0, 0, 0));
    place('modern_arm_chair_01', SPOT.seat.clone().setY(0).add(new THREE.Vector3(0, 0, 0.12)), Math.PI);
    place('desk_lamp_arm_01', new THREE.Vector3(-0.82, DESK_TOP, -0.28), 0.9);
    place('office_notepads', new THREE.Vector3(-0.45, DESK_TOP, 0.12), -0.35);
    place('potted_plant_02', new THREE.Vector3(-1.45, 0, -0.35), 0.4, 1.2);
  }

  #loadRobot() {
    new GLTFLoader().load('../assets/RobotExpressive.glb', (gltf) => {
      const robot = gltf.scene;
      const box = new THREE.Box3().setFromObject(robot);
      robot.scale.setScalar(ROBOT_HEIGHT / (box.max.y - box.min.y));
      robot.position.copy(SPOT.stand);
      robot.rotation.y = HEADING.you;
      robot.traverse((o) => {
        if (o.isMesh) {
          o.castShadow = true;
          o.receiveShadow = true;
          if (o.material?.isMeshStandardMaterial) o.material.envMapIntensity = 1.2;
        }
      });
      this.scene.add(robot);
      this.robot = robot;
      this.heading = HEADING.you;
      this.bones = {};
      robot.traverse((o) => { if (o.isBone) this.bones[o.name] = o; });

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
      // a head-only nod that can play on top of sitting
      const yes = gltf.animations.find((c) => c.name === 'Yes');
      if (yes) {
        const head = yes.clone();
        head.tracks = head.tracks.filter((t) => /^(Head|Neck)\./.test(t.name));
        THREE.AnimationUtils.makeClipAdditive(head);
        this.nodAction = this.mixer.clipAction(head);
        this.nodAction.loop = THREE.LoopOnce;
        this.nodAction.clampWhenFinished = false;
      }
      this.current = this.actions.Idle;
      this.current.play();
      this.mixer.addEventListener('finished', (e) => { if (e.action === this.current && this.after) this.after(); });
      window.__robotClips = Object.keys(this.actions);
      window.__bones = Object.keys(this.bones);
    });
  }

  #buildComposer() {
    const target = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
    const composer = new EffectComposer(this.renderer, target);
    composer.addPass(new RenderPass(this.scene, this.camera));
    this.gtao = new GTAOPass(this.scene, this.camera, 1, 1);
    this.gtao.updateGtaoMaterial({ radius: 0.35, distanceFallOff: 1, thickness: 1 });
    this.gtao.blendIntensity = 0.85;
    composer.addPass(this.gtao);
    this.bokeh = new BokehPass(this.scene, this.camera, { focus: SHOTS.idle.focus, aperture: 0.0016, maxblur: 0.0055 });
    composer.addPass(this.bokeh);
    this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.2, 0.5, 0.97);
    composer.addPass(this.bloom);
    composer.addPass(new FilmPass(0.12, false));
    composer.addPass(new OutputPass());
    this.composer = composer;
  }

  // --- the robot's choreography ---------------------------------------------

  #fadeTo(name, seconds = 0.35) {
    const action = this.actions?.[name];
    if (!action) return;
    if (action === this.current) {
      if (action.loop === THREE.LoopOnce) action.reset().play();
      return;
    }
    this.current?.fadeOut(seconds);
    action.reset().setEffectiveWeight(1).fadeIn(seconds).play();
    this.current = action;
  }

  #emote(name, then = 'Idle') {
    if (!this.actions || this.seated) return;
    this.#fadeTo(name, 0.2);
    this.after = () => { this.after = null; this.#fadeTo(then, 0.4); };
  }

  #nod() {
    if (!this.nodAction) return;
    this.nodAction.reset().setEffectiveWeight(1.4).play();
  }

  // a move: walk from where it is to `to`, turning to `heading`, then run `then`
  #walk(to, heading, then) {
    if (!this.robot) return;
    this.move = { from: this.robot.position.clone(), to: to.clone(), t: 0, dur: Math.max(0.6, this.robot.position.distanceTo(to) / 0.75), heading, then };
    this.after = null;
    this.#fadeTo('Walking', 0.3);
  }

  #goSit() {
    if (!this.robot || this.seated || this.sitting) return;
    this.sitting = true;
    this.#walk(SPOT.chairFront, HEADING.desk, () => {
      this.move = { from: this.robot.position.clone(), to: SPOT.seat.clone(), t: 0, dur: 0.7, heading: HEADING.desk, then: () => { this.seated = true; this.sitting = false; } };
      this.#fadeTo('Sitting', 0.5);
    });
  }

  #goStand(state) {
    if (!this.robot) return;
    const finish = () => {
      if (state === 'done') this.#emote('ThumbsUp', 'Idle');
      else if (state === 'stopped' || state === 'error') this.#emote('No', 'Idle');
    };
    if (!this.seated && !this.sitting) {
      this.heading = HEADING.you;
      finish();
      return;
    }
    this.seated = false;
    this.sitting = false;
    this.#fadeTo('Idle', 0.5);
    this.move = {
      from: this.robot.position.clone(), to: SPOT.chairFront.clone(), t: 0, dur: 0.6, heading: HEADING.desk,
      then: () => this.#walk(SPOT.stand, HEADING.you, () => { this.#fadeTo('Idle', 0.3); finish(); }),
    };
  }

  #updateMove(dt) {
    const m = this.move;
    if (!m || !this.robot) return;
    m.t = Math.min(1, m.t + dt / m.dur);
    const k = ease(m.t);
    this.robot.position.lerpVectors(m.from, m.to, k);
    const dir = m.to.clone().sub(m.from);
    const walking = this.current === this.actions?.Walking && dir.lengthSq() > 0.01;
    const want = walking && m.t < 0.85 ? Math.atan2(dir.x, dir.z) : m.heading;
    this.heading = want;
    if (m.t >= 1) {
      this.move = null;
      this.heading = m.heading;
      m.then?.();
    }
  }

  // procedural layer on top of the clips: where the head looks, typing, the mouse arm
  #updateBones(dt, t) {
    const b = this.bones;
    if (!b?.Head) return;
    this.typing = Math.max(0, this.typing - dt * 0.9);
    this.mousing = Math.max(0, this.mousing - dt * 1.2);
    this.thinking = Math.max(0, this.thinking - dt * 0.35);
    const working = this.seated;
    // head: toward the point on the monitor while working, a slow look-around otherwise
    const yaw = working ? (0.5 - this.pointer.x) * 0.5 + Math.sin(t * 0.7) * 0.02 : Math.sin(t * 0.35) * 0.18;
    const pitch = working ? (this.pointer.y - 0.5) * 0.25 - 0.05 + this.thinking * -0.12 : Math.sin(t * 0.5) * 0.05;
    const roll = this.thinking * 0.16 * Math.sin(t * 1.3);
    this.headAim = this.headAim ?? new THREE.Euler();
    this.headAim.set(damp(this.headAim.x, pitch, 6, dt), damp(this.headAim.y, yaw, 6, dt), damp(this.headAim.z, roll, 4, dt));
    b.Head.quaternion.multiply(new THREE.Quaternion().setFromEuler(this.headAim));
    if (!working) return;
    // typing: both forearms tap; mouse: the right arm sweeps
    const tap = this.typing * 0.12;
    if (b['LowerArm.L']) b['LowerArm.L'].rotateX(tap * Math.sin(t * 19));
    if (b['LowerArm.R']) b['LowerArm.R'].rotateX(tap * Math.sin(t * 17 + 1.3));
    if (b['UpperArm.R']) b['UpperArm.R'].rotateZ(this.mousing * 0.12 * Math.sin(t * 6));
  }

  // --- camera ------------------------------------------------------------------

  #updateCamera(dt, t) {
    const shot = this.state === 'working' || this.state === 'planning' ? SHOTS.working
      : this.state === 'done' ? SHOTS.done : SHOTS.idle;
    const sway = new THREE.Vector3(Math.sin(t * 0.21) * 0.05, Math.sin(t * 0.33) * 0.025, Math.cos(t * 0.17) * 0.04);
    const manual = t < this.manualUntil;
    if (!manual) {
      const pos = new THREE.Vector3(...shot.pos).add(sway);
      this.camera.position.lerp(pos, 1 - Math.exp(-1.1 * dt));
      this.lookAt.lerp(new THREE.Vector3(...shot.target), 1 - Math.exp(-1.6 * dt));
      this.controls.target.copy(this.lookAt);
    }
    this.controls.update();
    const focus = this.camera.position.distanceTo(this.robot ? this.robot.position.clone().setY(0.9) : this.lookAt);
    this.bokeh.uniforms.focus.value = damp(this.bokeh.uniforms.focus.value, focus, 3, dt);
  }

  #frame() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    this.#updateMove(dt);
    this.mixer?.update(dt);
    if (this.robot) {
      const diff = Math.atan2(Math.sin(this.heading - this.robot.rotation.y), Math.cos(this.heading - this.robot.rotation.y));
      this.robot.rotation.y += diff * Math.min(1, dt * 5);
    }
    this.#updateBones(dt, t);
    this.#updateCamera(dt, t);
    this.screenLight.intensity = 4 + (this.state === 'working' ? 1.5 + Math.sin(t * 3) * 0.4 : 0);
    if (this.screen.point && t - this.screen.point.t < 1.2) this.#drawScreen();
    else if (Math.floor(t * 2) !== this.blink) { this.blink = Math.floor(t * 2); this.#drawScreen(); }
    this.composer.render(dt);
  }

  #resize() {
    const { innerWidth: w, innerHeight: h } = window;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  // --- the monitor -------------------------------------------------------------

  #drawScreen() {
    const c = this.screenCanvas?.getContext('2d');
    if (!c) return;
    const { width: W, height: H } = this.screenCanvas;
    const s = this.screen;
    const accent = STATE_COLOR[s.status] ?? STATE_COLOR.idle;
    const now = this.clock.elapsedTime;

    if (s.image) {
      // the real screen the agent is working on, with where it pointed last
      const r = Math.max(W / s.image.width, H / s.image.height);
      const w = s.image.width * r;
      const h = s.image.height * r;
      c.drawImage(s.image, (W - w) / 2, (H - h) / 2, w, h);
      if (s.point) {
        const x = (W - w) / 2 + s.point.x * w;
        const y = (H - h) / 2 + s.point.y * h;
        const age = now - s.point.t;
        const color = CHANNEL_COLOR[s.point.channel] ?? accent;
        c.strokeStyle = color;
        c.lineWidth = 4;
        c.beginPath();
        c.arc(x, y, 14 + Math.min(age, 1.2) * 40, 0, Math.PI * 2);
        c.globalAlpha = Math.max(0, 1 - age / 1.2);
        c.stroke();
        c.globalAlpha = 1;
        c.beginPath();
        c.arc(x, y, 9, 0, Math.PI * 2);
        c.fillStyle = color;
        c.fill();
      }
    } else {
      const g = c.createLinearGradient(0, 0, W, H);
      g.addColorStop(0, '#0c1016');
      g.addColorStop(1, '#151b24');
      c.fillStyle = g;
      c.fillRect(0, 0, W, H);
      c.font = '500 34px "JetBrains Mono", Consolas, monospace';
      c.fillStyle = '#c9d1d9';
      c.fillText(s.goal ? s.goal.slice(0, 56) : 'waiting for a command', 60, 120);
      c.fillStyle = '#6b7480';
      c.font = '28px "JetBrains Mono", Consolas, monospace';
      s.lines.slice(-12).forEach((line, i) => c.fillText(`${String(line.n).padStart(2)}  ${line.text}`.slice(0, 70), 60, 190 + i * 40));
      if (s.status === 'working' && this.blink % 2 === 0) {
        c.fillStyle = accent;
        c.fillRect(60, 190 + Math.min(s.lines.length, 12) * 40 - 26, 16, 30);
      }
    }
    // status strip along the top
    c.fillStyle = 'rgba(8, 10, 14, 0.82)';
    c.fillRect(0, 0, W, 54);
    c.beginPath();
    c.arc(34, 27, 9, 0, Math.PI * 2);
    c.fillStyle = accent;
    c.fill();
    c.font = '600 26px "Inter Variable", "Segoe UI", sans-serif';
    c.fillStyle = '#e8ecf1';
    c.fillText(s.app ? `agent · ${s.app}` : 'agent', 56, 36);
    c.textAlign = 'right';
    c.fillStyle = accent;
    c.fillText(s.status, W - 28, 36);
    c.textAlign = 'left';
    this.screenTexture.needsUpdate = true;
  }
}
