/**
 * GRPO (Group Relative Policy Optimization) Algorithm
 * Implements policy creation and action selection
 * @typedef {import('@tensorflow/tfjs').Sequential} Sequential
 * @typedef {import('@tensorflow/tfjs').Tensor} Tensor
 */

/**
 * Creates a neural network policy for GRPO
 * @returns {Sequential} TensorFlow.js sequential model
 * @throws {Error} If TensorFlow.js is not loaded
 */
function createPolicy() {
  if (typeof tf === 'undefined') {
    throw new Error('TensorFlow.js is not loaded. Please ensure the CDN script is loaded.');
  }
  const m = tf.sequential();
  m.add(tf.layers.dense({ units: 48, activation: 'tanh', inputShape: [18], kernelInitializer: 'glorotNormal' }));
  m.add(tf.layers.dense({ units: 32, activation: 'tanh' }));
  m.add(tf.layers.dense({ units: 12 })); // mean(6) + logStd(6)
  return m;
}

/**
 * Gets an action from the policy given the current state
 * @param {Sequential} policy - The policy network
 * @param {number[]} state - Current environment state (18 values)
 * @param {boolean} [explore=true] - Whether to add exploration noise
 * @returns {number[]} Array of 6 action values in range [-1, 1]
 */
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

/**
 * Generates a random number from standard normal distribution
 * @returns {number} Random value from N(0,1)
 */
function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
