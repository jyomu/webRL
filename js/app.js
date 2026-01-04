/**
 * Main Application Logic
 * Handles UI, rendering, and coordination between components
 */

const $ = id => document.getElementById(id);
const P = id => parseFloat($(id).value);

// Helper function for environment
window.getParameter = P;

const canvas = $('sim');
const ctx = canvas.getContext('2d');

// Worker must be declared before any code that might call sendParams()
let worker = null;
let workerReady = false;

// Setup slider controls
document.querySelectorAll('.row').forEach(row => {
  const inp = row.querySelector('input');
  const sp = row.querySelector('span');
  if (!inp || !sp) return;
  inp.oninput = () => {
    let v = parseFloat(inp.value);
    sp.textContent = inp.id === 'lr' ? `${Math.pow(10, v).toExponential(0)}` : v % 1 === 0 ? v : v.toFixed(2);
    sendParams();
  };
  inp.oninput();
});

// Initialize
let policy = createPolicy();
let biped = new Biped();
let training = false;
let testing = false;
let speedMult = 1;
let episode = 0;
let stepCount = 0;
let totReward = 0;
let lastLoss = 0;
let curActions = [0, 0, 0, 0, 0, 0];
let rewardHistory = [];
let lastT = performance.now();
let fps = 0;
let bestTraj = null;

function initWorker() {
  const workerCode = document.getElementById('workerScript').textContent;
  const blob = new Blob([workerCode], { type: 'application/javascript' });
  worker = new Worker(URL.createObjectURL(blob));
  setupWorkerHandlers();
  worker.postMessage({ type: 'init' });
  sendParams();
}

function setupWorkerHandlers() {
  worker.onmessage = (e) => {
  const { type, data } = e.data;
  if (type === 'ready') {
    workerReady = true;
    $('workerStatus').textContent = '●Ready';
    $('workerStatus').style.color = '#4f4';
    syncWeightsToWorker();
  } else if (type === 'trained') {
    lastLoss = data.loss;
    rewardHistory.push(data.avgR);
    if (rewardHistory.length > 100) rewardHistory.shift();
    bestTraj = data.bestTraj;
    episode++;

    // Sync weights
    const ws = data.weights.map(w => tf.tensor(w.data, w.shape));
    policy.setWeights(ws);
    ws.forEach(w => w.dispose());

    log(`EP${episode}: avg=${data.avgR.toFixed(1)} best=${data.bestR.toFixed(1)} d=${data.bestDist?.toFixed(0) || 0}`);
    drawChart();

    if (training) startTrainStep();
  } else if (type === 'weightsSet') {
    $('workerStatus').textContent = '●Synced';
  }
  };
}

function syncWeightsToWorker() {
  if (!worker) return;
  const ws = policy.getWeights().map(w => ({ data: Array.from(w.dataSync()), shape: w.shape }));
  worker.postMessage({ type: 'weights', data: ws });
}

function sendParams() {
  if (!worker) return;
  const p = {
    rVel: P('rVel'), rUp: P('rUp'), rH: P('rH'), rEff: P('rEff'), rFall: P('rFall'),
    torque: P('torque'), friction: P('friction')
  };
  worker.postMessage({ type: 'params', data: p });
}

function startTrainStep() {
  if (!training || !worker) return;
  $('workerStatus').textContent = '●Training';
  $('workerStatus').style.color = '#ff0';
  worker.postMessage({
    type: 'train', data: {
      G: Math.round(P('groupSize')),
      epLen: Math.round(P('epLen')),
      lr: Math.pow(10, P('lr')),
      ent: P('entropy')
    }
  });
}

// Initialize worker when script tag is populated with content
window.addEventListener('workerReady', () => {
  if (document.getElementById('workerScript').textContent.trim().length > 100) {
    initWorker();
  }
});

// UI Functions
function log(m) {
  const e = $('log');
  e.innerHTML = m + '<br>' + e.innerHTML.split('<br>').slice(0, 15).join('<br>');
}

function updateUI() {
  $('mEp').textContent = episode;
  $('mR').textContent = totReward.toFixed(1);
  $('mD').textContent = biped.dist().toFixed(0);
  $('mS').textContent = training ? '学習' : testing ? 'テスト' : biped.fallen ? '転倒' : '実行';
  $('mL').textContent = lastLoss.toFixed(4);
  $('mF').textContent = fps;

  const st = biped.getState();
  const nm = ['θ', 'ω', 'pθ', 'vx', 'vy', 'h', 'Lc', 'Rc', 'Lh', 'Lk', 'La', 'Lhv', 'Lkv', 'Rh', 'Rk', 'Ra', 'Rhv', 'Rkv'];
  let html = st.map((v, i) => `<div>${nm[i]}:${v.toFixed(1)}</div>`).join('');
  html += curActions.map((v, i) => `<div style="color:#0ff">${['LH', 'LK', 'LA', 'RH', 'RK', 'RA'][i]}:${v.toFixed(2)}</div>`).join('');
  $('stateGrid').innerHTML = html;
}

function drawChart() {
  const c = $('chart');
  const cx = c.getContext('2d');
  c.width = c.offsetWidth;
  c.height = 45;
  cx.fillStyle = '#111';
  cx.fillRect(0, 0, c.width, c.height);
  if (rewardHistory.length < 2) return;
  const min = Math.min(...rewardHistory);
  const max = Math.max(...rewardHistory);
  const range = max - min || 1;
  cx.strokeStyle = '#4f8';
  cx.lineWidth = 1.5;
  cx.beginPath();
  rewardHistory.forEach((r, i) => {
    const x = (i / (rewardHistory.length - 1)) * c.width;
    const y = c.height - 3 - ((r - min) / range) * (c.height - 6);
    i === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y);
  });
  cx.stroke();
  cx.fillStyle = '#aaa';
  cx.font = '8px monospace';
  cx.fillText(`${min.toFixed(0)}`, 2, c.height - 2);
  cx.fillText(`${max.toFixed(0)}`, 2, 10);
}

function render() {
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  ctx.fillStyle = '#1a2a1a';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  biped.render(ctx, canvas.width, canvas.height);
  ctx.fillStyle = '#fff';
  ctx.font = '12px monospace';
  ctx.fillText(`EP:${episode} R:${totReward.toFixed(1)} D:${biped.dist().toFixed(0)} Step:${stepCount}`, 10, 18);
  if (training) {
    ctx.fillStyle = '#4f4';
    ctx.fillText('● LEARNING (Worker)', 10, 34);
  }
  if (testing) {
    ctx.fillStyle = '#48f';
    ctx.fillText('● TESTING', 10, 34);
  }
}

function getParams() {
  return {
    rVel: P('rVel'), rUp: P('rUp'), rH: P('rH'), rEff: P('rEff'), rFall: P('rFall'),
    torque: P('torque'), friction: P('friction')
  };
}

// Main loop
let trajIdx = 0;
function mainLoop() {
  const now = performance.now();
  fps = Math.round(1000 / (now - lastT));
  lastT = now;

  // Best trajectory replay mode
  if (training && bestTraj && bestTraj.length > 0) {
    if (trajIdx >= bestTraj.length || biped.fallen) {
      biped.reset();
      totReward = 0;
      stepCount = 0;
      trajIdx = 0;
    }
    for (let i = 0; i < speedMult && trajIdx < bestTraj.length; i++) {
      const t = bestTraj[trajIdx++];
      curActions = t.actions;
      const { reward } = biped.step(t.actions, getParams());
      totReward += reward;
      stepCount++;
    }
  } else {
    // Normal/test mode
    if (biped.fallen || stepCount > 600) {
      biped.reset();
      totReward = 0;
      stepCount = 0;
    }
    for (let i = 0; i < speedMult; i++) {
      const state = biped.getState();
      const actions = getAction(policy, state, !testing);
      curActions = actions;
      const { reward } = biped.step(actions, getParams());
      totReward += reward;
      stepCount++;
      if (biped.fallen) break;
    }
  }

  render();
  updateUI();
  requestAnimationFrame(mainLoop);
}

// Button handlers
$('btnTrain').onclick = () => {
  training = !training;
  testing = false;
  $('btnTrain').textContent = training ? '学習停止' : '学習開始';
  $('btnTrain').className = training ? 'stop' : '';
  $('btnTest').className = '';
  if (training) {
    biped.reset();
    stepCount = 0;
    totReward = 0;
    trajIdx = 0;
    bestTraj = null;
    sendParams();
    syncWeightsToWorker();
    setTimeout(startTrainStep, 100);
  }
};

$('btnTest').onclick = () => {
  testing = !testing;
  training = false;
  $('btnTest').textContent = testing ? '探索ON' : 'テスト';
  $('btnTest').className = testing ? 'stop' : '';
  $('btnTrain').textContent = '学習開始';
  $('btnTrain').className = '';
  biped.reset();
  stepCount = 0;
  totReward = 0;
  bestTraj = null;
};

$('btnReset').onclick = () => {
  biped.reset();
  totReward = 0;
  stepCount = 0;
  trajIdx = 0;
};

$('btnSpeed').onclick = () => {
  speedMult = speedMult === 1 ? 3 : speedMult === 3 ? 6 : 1;
  $('btnSpeed').textContent = speedMult + 'x';
};

log('GRPO二足歩行 (Worker並列) 準備完了');
mainLoop();
