/**
 * Web Worker for parallel GRPO training
 * Runs training episodes in the background
 * 
 * Note: This file contains some code duplication with js/grpo.js and js/environment.js
 * This is intentional as Web Workers run in isolation and cannot easily import ES6 modules.
 * The duplicated code includes:
 * - Biped class (simplified version without rendering)
 * - randn() function
 * - createPolicy() function
 */

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.17.0');
importScripts('https://cdn.jsdelivr.net/npm/matter-js@0.19.0/build/matter.min.js');

const { Engine, Bodies, Body, Composite, Constraint } = Matter;
let policy = null;
let params = {};

class Biped {
  constructor(p) {
    this.params = p;
    this.engine = Engine.create({ gravity: { x: 0, y: 1 } });
    this.world = this.engine.world;
    this.groundY = 380;
    this.build();
  }

  build() {
    Composite.clear(this.world);
    const fr = this.params.friction || 1;
    const x = 200;
    const gy = this.groundY;
    this.ground = Bodies.rectangle(5000, gy + 30, 12000, 60, { isStatic: true, friction: fr });
    this.head = Bodies.circle(x, gy - 145, 12, { friction: fr, density: 0.001 });
    this.torso = Bodies.rectangle(x, gy - 115, 24, 45, { friction: fr, density: 0.002 });
    this.pelvis = Bodies.rectangle(x, gy - 82, 28, 18, { friction: fr, density: 0.002 });
    this.lThigh = Bodies.rectangle(x - 8, gy - 55, 14, 40, { friction: fr, density: 0.0015 });
    this.lShin = Bodies.rectangle(x - 8, gy - 18, 12, 38, { friction: fr, density: 0.001 });
    this.lFoot = Bodies.rectangle(x - 8, gy - 2, 22, 6, { friction: fr * 1.5, density: 0.001 });
    this.rThigh = Bodies.rectangle(x + 8, gy - 55, 14, 40, { friction: fr, density: 0.0015 });
    this.rShin = Bodies.rectangle(x + 8, gy - 18, 12, 38, { friction: fr, density: 0.001 });
    this.rFoot = Bodies.rectangle(x + 8, gy - 2, 22, 6, { friction: fr * 1.5, density: 0.001 });
    const st = 0.95;
    Composite.add(this.world, [
      this.ground, this.head, this.torso, this.pelvis,
      this.lThigh, this.lShin, this.lFoot, this.rThigh, this.rShin, this.rFoot,
      Constraint.create({ bodyA: this.head, pointA: { x: 0, y: 10 }, bodyB: this.torso, pointB: { x: 0, y: -20 }, stiffness: st, length: 0 }),
      Constraint.create({ bodyA: this.torso, pointA: { x: 0, y: 20 }, bodyB: this.pelvis, pointB: { x: 0, y: -6 }, stiffness: st, length: 0 }),
      Constraint.create({ bodyA: this.pelvis, pointA: { x: -6, y: 6 }, bodyB: this.lThigh, pointB: { x: 0, y: -18 }, stiffness: st, length: 0 }),
      Constraint.create({ bodyA: this.lThigh, pointA: { x: 0, y: 18 }, bodyB: this.lShin, pointB: { x: 0, y: -16 }, stiffness: st, length: 0 }),
      Constraint.create({ bodyA: this.lShin, pointA: { x: 0, y: 16 }, bodyB: this.lFoot, pointB: { x: 0, y: 0 }, stiffness: st, length: 0 }),
      Constraint.create({ bodyA: this.pelvis, pointA: { x: 6, y: 6 }, bodyB: this.rThigh, pointB: { x: 0, y: -18 }, stiffness: st, length: 0 }),
      Constraint.create({ bodyA: this.rThigh, pointA: { x: 0, y: 18 }, bodyB: this.rShin, pointB: { x: 0, y: -16 }, stiffness: st, length: 0 }),
      Constraint.create({ bodyA: this.rShin, pointA: { x: 0, y: 16 }, bodyB: this.rFoot, pointB: { x: 0, y: 0 }, stiffness: st, length: 0 })
    ]);
    this.initX = x;
    this.fallen = false;
  }

  reset() {
    this.build();
    return this.getState();
  }

  getState() {
    const t = this.torso;
    const p = this.pelvis;
    const gy = this.groundY;
    return [
      t.angle, t.angularVelocity * 2, p.angle, t.velocity.x / 5, t.velocity.y / 5, (gy - t.position.y) / 100,
      this.lFoot.position.y > gy - 8 ? 1 : -1, this.rFoot.position.y > gy - 8 ? 1 : -1,
      this.lThigh.angle - p.angle, this.lShin.angle - this.lThigh.angle, this.lFoot.angle,
      this.lThigh.angularVelocity, this.lShin.angularVelocity,
      this.rThigh.angle - p.angle, this.rShin.angle - this.rThigh.angle, this.rFoot.angle,
      this.rThigh.angularVelocity, this.rShin.angularVelocity
    ];
  }

  step(actions) {
    if (this.fallen) return { state: this.getState(), reward: -this.params.rFall, done: true };
    const tq = this.params.torque;
    const [lh, lk, la, rh, rk, ra] = actions;
    Body.setAngularVelocity(this.lThigh, this.lThigh.angularVelocity + lh * tq);
    Body.setAngularVelocity(this.lShin, this.lShin.angularVelocity + lk * tq);
    Body.setAngularVelocity(this.lFoot, this.lFoot.angularVelocity + la * tq * 0.5);
    Body.setAngularVelocity(this.rThigh, this.rThigh.angularVelocity + rh * tq);
    Body.setAngularVelocity(this.rShin, this.rShin.angularVelocity + rk * tq);
    Body.setAngularVelocity(this.rFoot, this.rFoot.angularVelocity + ra * tq * 0.5);
    Engine.update(this.engine, 1000 / 60);
    const state = this.getState();
    if (this.torso.position.y > this.groundY - 30 || Math.abs(this.torso.angle) > 1.3 || this.head.position.y > this.groundY - 40) {
      this.fallen = true;
      return { state, reward: -this.params.rFall, done: true };
    }
    let r = this.torso.velocity.x * this.params.rVel + (1 - Math.abs(this.torso.angle)) * this.params.rUp * 0.1;
    r += Math.min(1, (this.groundY - this.torso.position.y) / 90) * this.params.rH * 0.1;
    r -= actions.reduce((s, a) => s + a * a, 0) * this.params.rEff + 0.02;
    return { state, reward: r, done: false };
  }

  dist() {
    return this.torso.position.x - this.initX;
  }
}

function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function createPolicy() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 48, activation: 'tanh', inputShape: [18], kernelInitializer: 'glorotNormal' }));
  m.add(tf.layers.dense({ units: 32, activation: 'tanh' }));
  m.add(tf.layers.dense({ units: 12 }));
  return m;
}

function getActionWithInfo(state) {
  return tf.tidy(() => {
    const out = policy.predict(tf.tensor2d([state]));
    const [mean, logStd] = tf.split(out, 2, 1);
    const std = tf.exp(tf.clipByValue(logStd, -2, 0.5));
    const m = mean.dataSync();
    const s = std.dataSync();
    const a = [];
    for (let i = 0; i < 6; i++) a.push(Math.max(-1, Math.min(1, m[i] + randn() * s[i])));
    return { actions: a, mean: [...m], std: [...s] };
  });
}

function runEpisode(maxSteps) {
  const env = new Biped(params);
  env.reset();
  const traj = [];
  let totR = 0;
  for (let t = 0; t < maxSteps; t++) {
    const state = env.getState();
    const { actions, mean, std } = getActionWithInfo(state);
    const { reward, done } = env.step(actions);
    totR += reward;
    traj.push({ state, actions, mean, std });
    if (done) break;
  }
  return { traj, totR, dist: env.dist() };
}

async function train(G, epLen, lr, ent) {
  const samples = [];
  for (let i = 0; i < G; i++) samples.push(runEpisode(epLen));

  const rewards = samples.map(s => s.totR);
  const mean = rewards.reduce((a, b) => a + b, 0) / G;
  const std = Math.sqrt(rewards.reduce((a, b) => a + (b - mean) ** 2, 0) / G) + 1e-8;
  const advs = rewards.map(r => (r - mean) / std);

  const states = [], actions = [], oldMeans = [], oldStds = [], advArr = [];
  samples.forEach((s, idx) => {
    s.traj.forEach(t => {
      states.push(t.state);
      actions.push(t.actions);
      oldMeans.push(t.mean);
      oldStds.push(t.std);
      advArr.push(advs[idx]);
    });
  });

  if (!states.length) return { loss: 0, avgR: 0, bestR: 0, bestTraj: [] };

  const opt = tf.train.adam(lr);
  const loss = tf.tidy(() => {
    const stateT = tf.tensor2d(states);
    const actionT = tf.tensor2d(actions);
    const advT = tf.tensor1d(advArr);
    const oldMeanT = tf.tensor2d(oldMeans);
    const oldStdT = tf.tensor2d(oldStds);
    return opt.minimize(() => {
      const out = policy.predict(stateT);
      const [newMean, logStd] = tf.split(out, 2, 1);
      const newStd = tf.exp(tf.clipByValue(logStd, -2, 0.5));
      const diff = tf.sub(actionT, newMean);
      const logpNew = tf.sum(tf.sub(tf.mul(-0.5, tf.square(tf.div(diff, tf.add(newStd, 1e-8)))), tf.log(tf.add(newStd, 1e-8))), 1);
      const oldDiff = tf.sub(actionT, oldMeanT);
      const logpOld = tf.sum(tf.sub(tf.mul(-0.5, tf.square(tf.div(oldDiff, tf.add(oldStdT, 1e-8)))), tf.log(tf.add(oldStdT, 1e-8))), 1);
      const ratio = tf.exp(tf.sub(logpNew, logpOld));
      const clipped = tf.clipByValue(ratio, 0.8, 1.2);
      const pLoss = tf.neg(tf.mean(tf.minimum(tf.mul(ratio, advT), tf.mul(clipped, advT))));
      const entropyB = tf.mean(tf.sum(tf.log(tf.add(newStd, 1e-8)), 1));
      return tf.add(pLoss, tf.mul(-ent, entropyB));
    }, true).dataSync()[0];
  });
  opt.dispose();

  const best = samples.reduce((a, b) => a.totR > b.totR ? a : b);
  return { loss, avgR: mean, bestR: Math.max(...rewards), bestTraj: best.traj, bestDist: best.dist };
}

self.onmessage = async (e) => {
  const { type, data } = e.data;
  if (type === 'init') {
    policy = createPolicy();
    self.postMessage({ type: 'ready' });
  } else if (type === 'params') {
    params = data;
  } else if (type === 'weights') {
    const ws = data.map(w => tf.tensor(w.data, w.shape));
    policy.setWeights(ws);
    self.postMessage({ type: 'weightsSet' });
  } else if (type === 'train') {
    const { G, epLen, lr, ent } = data;
    const res = await train(G, epLen, lr, ent);
    const ws = policy.getWeights().map(w => ({ data: Array.from(w.dataSync()), shape: w.shape }));
    self.postMessage({ type: 'trained', data: { ...res, weights: ws } });
  } else if (type === 'getWeights') {
    const ws = policy.getWeights().map(w => ({ data: Array.from(w.dataSync()), shape: w.shape }));
    self.postMessage({ type: 'weights', data: ws });
  }
};
