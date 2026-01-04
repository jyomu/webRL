/**
 * GRPO (Group Relative Policy Optimization) Algorithm
 * Implements policy creation and action selection
 */

function createPolicy() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 48, activation: 'tanh', inputShape: [18], kernelInitializer: 'glorotNormal' }));
  m.add(tf.layers.dense({ units: 32, activation: 'tanh' }));
  m.add(tf.layers.dense({ units: 12 })); // mean(6) + logStd(6)
  return m;
}

function getAction(policy, state, explore = true) {
  return tf.tidy(() => {
    const out = policy.predict(tf.tensor2d([state]));
    const [mean, logStd] = tf.split(out, 2, 1);
    const std = tf.exp(tf.clipByValue(logStd, -2, 0.5));
    const m = mean.dataSync();
    const s = std.dataSync();
    const a = [];
    for (let i = 0; i < 6; i++) {
      let v = m[i];
      if (explore) v += randn() * s[i];
      a.push(Math.max(-1, Math.min(1, v)));
    }
    return a;
  });
}

function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
