/**
 * Bipedal Walker Environment
 * Uses Matter.js for physics simulation
 */

class Biped {
  constructor() {
    const { Engine } = Matter;
    this.engine = Engine.create({ gravity: { x: 0, y: 1 } });
    this.world = this.engine.world;
    this.groundY = 380;
    this.build();
  }

  build() {
    const { Bodies, Body, Composite, Constraint } = Matter;
    Composite.clear(this.world);
    
    const friction = window.getParameter ? window.getParameter('friction') : 1.0;
    const x = 200;
    const gy = this.groundY;

    // Ground
    this.ground = Bodies.rectangle(5000, gy + 30, 12000, 60, { isStatic: true, friction });

    // Head
    this.head = Bodies.circle(x, gy - 145, 12, { friction, density: 0.001, label: 'head' });

    // Torso (upper body)
    this.torso = Bodies.rectangle(x, gy - 115, 24, 45, { friction, density: 0.002, label: 'torso' });

    // Pelvis
    this.pelvis = Bodies.rectangle(x, gy - 82, 28, 18, { friction, density: 0.002, label: 'pelvis' });

    // Left leg
    this.lThigh = Bodies.rectangle(x - 8, gy - 55, 14, 40, { friction, density: 0.0015, label: 'lThigh' });
    this.lShin = Bodies.rectangle(x - 8, gy - 18, 12, 38, { friction, density: 0.001, label: 'lShin' });
    this.lFoot = Bodies.rectangle(x - 8, gy - 2, 22, 6, { friction: friction * 1.5, density: 0.001, label: 'lFoot' });

    // Right leg
    this.rThigh = Bodies.rectangle(x + 8, gy - 55, 14, 40, { friction, density: 0.0015, label: 'rThigh' });
    this.rShin = Bodies.rectangle(x + 8, gy - 18, 12, 38, { friction, density: 0.001, label: 'rShin' });
    this.rFoot = Bodies.rectangle(x + 8, gy - 2, 22, 6, { friction: friction * 1.5, density: 0.001, label: 'rFoot' });

    // Joints - Higher stiffness and damping to prevent joint disconnection
    const stiff = 1.0;
    const damp = 0.1;
    this.neck = Constraint.create({ bodyA: this.head, pointA: { x: 0, y: 10 }, bodyB: this.torso, pointB: { x: 0, y: -20 }, stiffness: stiff, damping: damp, length: 0 });
    this.spine = Constraint.create({ bodyA: this.torso, pointA: { x: 0, y: 20 }, bodyB: this.pelvis, pointB: { x: 0, y: -6 }, stiffness: stiff, damping: damp, length: 0 });

    this.lHip = Constraint.create({ bodyA: this.pelvis, pointA: { x: -6, y: 6 }, bodyB: this.lThigh, pointB: { x: 0, y: -18 }, stiffness: stiff, damping: damp, length: 0 });
    this.lKnee = Constraint.create({ bodyA: this.lThigh, pointA: { x: 0, y: 18 }, bodyB: this.lShin, pointB: { x: 0, y: -16 }, stiffness: stiff, damping: damp, length: 0 });
    this.lAnkle = Constraint.create({ bodyA: this.lShin, pointA: { x: 0, y: 16 }, bodyB: this.lFoot, pointB: { x: 0, y: 0 }, stiffness: stiff, damping: damp, length: 0 });

    this.rHip = Constraint.create({ bodyA: this.pelvis, pointA: { x: 6, y: 6 }, bodyB: this.rThigh, pointB: { x: 0, y: -18 }, stiffness: stiff, damping: damp, length: 0 });
    this.rKnee = Constraint.create({ bodyA: this.rThigh, pointA: { x: 0, y: 18 }, bodyB: this.rShin, pointB: { x: 0, y: -16 }, stiffness: stiff, damping: damp, length: 0 });
    this.rAnkle = Constraint.create({ bodyA: this.rShin, pointA: { x: 0, y: 16 }, bodyB: this.rFoot, pointB: { x: 0, y: 0 }, stiffness: stiff, damping: damp, length: 0 });

    Composite.add(this.world, [
      this.ground, this.head, this.torso, this.pelvis,
      this.lThigh, this.lShin, this.lFoot, this.rThigh, this.rShin, this.rFoot,
      this.neck, this.spine, this.lHip, this.lKnee, this.lAnkle, this.rHip, this.rKnee, this.rAnkle
    ]);

    this.initX = x;
    this.fallen = false;
    this.steps = 0;
  }

  reset() {
    this.build();
    return this.getState();
  }

  getState() {
    const t = this.torso;
    const p = this.pelvis;
    const h = (this.groundY - t.position.y) / 100;
    const vx = t.velocity.x / 5;
    const vy = t.velocity.y / 5;
    const ang = t.angle;
    const angV = t.angularVelocity * 2;
    const pAng = p.angle;

    const lha = this.lThigh.angle - p.angle;
    const lka = this.lShin.angle - this.lThigh.angle;
    const laa = this.lFoot.angle;
    const rha = this.rThigh.angle - p.angle;
    const rka = this.rShin.angle - this.rThigh.angle;
    const raa = this.rFoot.angle;
    const lhv = this.lThigh.angularVelocity;
    const lkv = this.lShin.angularVelocity;
    const rhv = this.rThigh.angularVelocity;
    const rkv = this.rShin.angularVelocity;

    const lc = this.lFoot.position.y > this.groundY - 8 ? 1 : -1;
    const rc = this.rFoot.position.y > this.groundY - 8 ? 1 : -1;

    return [ang, angV, pAng, vx, vy, h, lc, rc, lha, lka, laa, lhv, lkv, rha, rka, raa, rhv, rkv];
  }

  step(actions, params) {
    if (this.fallen) return { state: this.getState(), reward: -params.rFall, done: true };

    const { Body, Engine } = Matter;
    const tq = params.torque;
    const [lh, lk, la, rh, rk, ra] = actions;

    // Apply torque with gentle limits to prevent extreme movements while maintaining natural motion
    const MAX_ANGULAR_VELOCITY = 0.5; // rad/s - Prevents unrealistic fast rotations
    
    const applyTorqueWithLimit = (body, action, multiplier = 1.0) => {
      // Calculate new angular velocity (similar to original but with limits)
      const newAngVel = body.angularVelocity + action * tq * multiplier;
      // Clamp to prevent extreme rotations
      const clampedVel = Math.max(-MAX_ANGULAR_VELOCITY, Math.min(MAX_ANGULAR_VELOCITY, newAngVel));
      Body.setAngularVelocity(body, clampedVel);
    };
    
    applyTorqueWithLimit(this.lThigh, lh);
    applyTorqueWithLimit(this.lShin, lk);
    applyTorqueWithLimit(this.lFoot, la, 0.5);
    applyTorqueWithLimit(this.rThigh, rh);
    applyTorqueWithLimit(this.rShin, rk);
    applyTorqueWithLimit(this.rFoot, ra, 0.5);

    Engine.update(this.engine, 1000 / 60);
    this.steps++;

    const state = this.getState();
    const tooLow = this.torso.position.y > this.groundY - 30;
    const tooTilt = Math.abs(this.torso.angle) > 1.3;
    const headLow = this.head.position.y > this.groundY - 40;

    if (tooLow || tooTilt || headLow) {
      this.fallen = true;
      return { state, reward: -params.rFall, done: true };
    }

    let r = 0;
    r += this.torso.velocity.x * params.rVel;
    r += (1 - Math.abs(this.torso.angle)) * params.rUp * 0.1;
    r += Math.min(1, (this.groundY - this.torso.position.y) / 90) * params.rH * 0.1;
    r -= actions.reduce((s, a) => s + a * a, 0) * params.rEff;
    r += 0.02; // Base reward to encourage survival

    return { state, reward: r, done: false };
  }

  dist() {
    return this.torso.position.x - this.initX;
  }

  render(ctx, w, h) {
    const { Vector } = Matter;
    const camX = w / 2 - this.torso.position.x;
    ctx.save();
    ctx.translate(camX, 0);

    // Ground
    ctx.fillStyle = '#2a3a2a';
    ctx.fillRect(-2000, this.groundY, 14000, 100);

    // Grid
    ctx.strokeStyle = '#3a4a3a';
    ctx.fillStyle = '#555';
    ctx.font = '9px monospace';
    for (let x = Math.floor((this.torso.position.x - w) / 100) * 100; x < this.torso.position.x + w; x += 100) {
      ctx.beginPath();
      ctx.moveTo(x, this.groundY);
      ctx.lineTo(x, 0);
      ctx.stroke();
      ctx.fillText(x, x + 2, this.groundY - 3);
    }

    const drawBody = (b, col) => {
      ctx.save();
      ctx.translate(b.position.x, b.position.y);
      ctx.rotate(b.angle);
      ctx.fillStyle = col;
      if (b.label === 'head') {
        ctx.beginPath();
        ctx.arc(0, 0, b.circleRadius, 0, Math.PI * 2);
        ctx.fill();
      } else {
        const v = b.vertices;
        const cx = b.position.x;
        const cy = b.position.y;
        ctx.beginPath();
        ctx.moveTo(v[0].x - cx, v[0].y - cy);
        for (let i = 1; i < v.length; i++) ctx.lineTo(v[i].x - cx, v[i].y - cy);
        ctx.closePath();
        ctx.fill();
      }
      ctx.restore();
    };

    const c = this.fallen ? '#833' : '#48a';
    drawBody(this.head, '#fa8');
    drawBody(this.torso, c);
    drawBody(this.pelvis, c);
    drawBody(this.lThigh, '#c55');
    drawBody(this.lShin, '#a66');
    drawBody(this.lFoot, '#844');
    drawBody(this.rThigh, '#5a5');
    drawBody(this.rShin, '#6a6');
    drawBody(this.rFoot, '#484');

    // Joints
    [this.lHip, this.lKnee, this.lAnkle, this.rHip, this.rKnee, this.rAnkle].forEach(c => {
      const p = Vector.add(c.bodyA.position, Vector.rotate(c.pointA, c.bodyA.angle));
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#ff0';
      ctx.fill();
    });

    ctx.restore();
  }
}
