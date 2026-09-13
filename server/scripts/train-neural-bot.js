const fs = require('fs');
const path = require('path');
const { INPUT_SIZE } = require('../src/game/ai/NeuralObservation');

const OUTPUT_SIZE = 6;
const STEPS = Number.parseInt(process.env.NEURAL_TRAIN_STEPS || '70000', 10);
const SEED = Number.parseInt(process.env.NEURAL_TRAIN_SEED || '20260912', 10);
const MODEL_PATH = path.resolve(__dirname, '../src/game/ai/models/neural-player-v1.json');
const TEMP_PATH = path.resolve(__dirname, '../.neural-training-tmp');

function rng(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let n = state;
    n = Math.imul(n ^ (n >>> 15), n | 1);
    n ^= n + Math.imul(n ^ (n >>> 7), n | 61);
    return ((n ^ (n >>> 14)) >>> 0) / 4294967296;
  };
}

const random = rng(SEED);
const between = (a, b) => a + (b - a) * random();

function unit(x, y) {
  const length = Math.hypot(x, y);
  return length > 1e-6 ? [x / length, y / length] : [1, 0];
}

function scenario() {
  const x = new Float32Array(INPUT_SIZE);
  x[0] = between(0.05, 1);
  x[1] = random() < 0.82 ? 1 : 0;
  x[2] = random() < 0.72 ? 1 : 0;
  x[3] = random() < 0.08 ? 1 : 0;

  if (random() < 0.7) {
    const a = between(-Math.PI, Math.PI), d = between(0.04, 1);
    x[4] = 1; x[5] = Math.cos(a) * d; x[6] = Math.sin(a) * d; x[7] = d;
    x[8] = between(0.05, 1);
    x[random() < 0.65 ? 9 : 10] = 1;
    x[11] = between(0.08, 0.8);
  }

  if (random() < 0.42) {
    const a = between(-Math.PI, Math.PI), d = between(0.02, 1);
    x[12] = 1; x[13] = Math.cos(a) * d; x[14] = Math.sin(a) * d; x[17] = d;
    let direction = between(-Math.PI, Math.PI);
    if (random() < 0.62) direction = a + Math.PI + between(-0.35, 0.35);
    x[15] = Math.cos(direction); x[16] = Math.sin(direction);
  }

  if (random() < 0.68) {
    const a = between(-Math.PI, Math.PI), d = between(0.03, 1);
    x[18] = 1; x[19] = Math.cos(a) * d; x[20] = Math.sin(a) * d; x[21] = d;
    x[22] = random() < 0.55 ? 1 : -1;
  }

  if (random() < 0.48) {
    const a = between(-Math.PI, Math.PI), d = between(0.02, 1);
    x[23] = 1; x[24] = Math.cos(a) * d; x[25] = Math.sin(a) * d; x[26] = d;
  }

  const phase = between(-Math.PI, Math.PI);
  x[27] = Math.sin(phase); x[28] = Math.cos(phase);
  return x;
}

function teacher(x) {
  let moveX = x[28], moveY = x[27];
  let aimX = moveX, aimY = moveY;
  let swing = -1, throwSword = -1;

  if (x[18]) {
    [moveX, moveY] = unit(x[19], x[20]);
    [aimX, aimY] = [moveX, moveY];
    if (x[22] < 0 && x[21] < 0.17 && x[1]) swing = 1;
  }

  if (x[4]) {
    [aimX, aimY] = unit(x[5], x[6]);
    if (x[0] < 0.3) {
      [moveX, moveY] = [-aimX, -aimY];
    } else if (x[7] > 0.23) {
      [moveX, moveY] = [aimX, aimY];
    } else {
      const side = x[27] >= 0 ? 1 : -1;
      [moveX, moveY] = [-aimY * side, aimX * side];
    }
    if (x[1] && x[7] < 0.18) swing = 1;
    if (x[2] && x[7] > 0.24 && x[7] < 0.72) throwSword = 1;
  }

  if (x[12] && x[17] < 0.48) {
    const incoming = x[15] * -x[13] + x[16] * -x[14];
    if (incoming > 0.42) {
      const side = (-x[16] * -x[13] + x[15] * -x[14]) >= 0 ? 1 : -1;
      moveX = -x[16] * side;
      moveY = x[15] * side;
    }
  }

  if (x[23] && x[26] < 0.3) {
    const [awayX, awayY] = unit(-x[24], -x[25]);
    [moveX, moveY] = unit(moveX + awayX * 1.6, moveY + awayY * 1.6);
  }

  [moveX, moveY] = unit(moveX, moveY);
  [aimX, aimY] = unit(aimX, aimY);
  if (x[3]) { swing = -1; throwSword = -1; }
  return Float32Array.from([moveX, moveY, aimX, aimY, swing, throwSword]);
}

function makeLayer(inputSize, outputSize) {
  const limit = Math.sqrt(6 / (inputSize + outputSize));
  const weights = new Float64Array(inputSize * outputSize);
  for (let i = 0; i < weights.length; i++) weights[i] = between(-limit, limit);
  return {
    inputSize, outputSize, weights, biases: new Float64Array(outputSize),
    mw: new Float64Array(weights.length), vw: new Float64Array(weights.length),
    mb: new Float64Array(outputSize), vb: new Float64Array(outputSize),
  };
}

const layers = [makeLayer(INPUT_SIZE, 32), makeLayer(32, 24), makeLayer(24, OUTPUT_SIZE)];

function forward(input) {
  const activations = [input];
  for (const layer of layers) {
    const previous = activations[activations.length - 1];
    const output = new Float64Array(layer.outputSize);
    for (let o = 0; o < layer.outputSize; o++) {
      let sum = layer.biases[o], row = o * layer.inputSize;
      for (let i = 0; i < layer.inputSize; i++) sum += layer.weights[row + i] * previous[i];
      output[o] = Math.tanh(sum);
    }
    activations.push(output);
  }
  return activations;
}

let adamStep = 0;
function trainOne(input, target) {
  const activations = forward(input);
  const deltas = new Array(layers.length);
  const last = activations[activations.length - 1];
  deltas[deltas.length - 1] = new Float64Array(OUTPUT_SIZE);
  let loss = 0;
  for (let o = 0; o < OUTPUT_SIZE; o++) {
    const error = last[o] - target[o];
    loss += error * error;
    deltas[deltas.length - 1][o] = (2 * error / OUTPUT_SIZE) * (1 - last[o] * last[o]);
  }

  for (let l = layers.length - 2; l >= 0; l--) {
    const current = layers[l], next = layers[l + 1];
    const activation = activations[l + 1];
    const delta = new Float64Array(current.outputSize);
    for (let i = 0; i < current.outputSize; i++) {
      let sum = 0;
      for (let o = 0; o < next.outputSize; o++) sum += next.weights[o * next.inputSize + i] * deltas[l + 1][o];
      delta[i] = sum * (1 - activation[i] * activation[i]);
    }
    deltas[l] = delta;
  }

  adamStep++;
  const lr = 0.0025, b1 = 0.9, b2 = 0.999, eps = 1e-8;
  const b1Correction = 1 - Math.pow(b1, adamStep);
  const b2Correction = 1 - Math.pow(b2, adamStep);
  for (let l = 0; l < layers.length; l++) {
    const layer = layers[l], previous = activations[l], delta = deltas[l];
    for (let o = 0; o < layer.outputSize; o++) {
      const gb = Math.max(-2, Math.min(2, delta[o]));
      layer.mb[o] = b1 * layer.mb[o] + (1 - b1) * gb;
      layer.vb[o] = b2 * layer.vb[o] + (1 - b2) * gb * gb;
      layer.biases[o] -= lr * (layer.mb[o] / b1Correction) / (Math.sqrt(layer.vb[o] / b2Correction) + eps);
      const row = o * layer.inputSize;
      for (let i = 0; i < layer.inputSize; i++) {
        const index = row + i;
        const gradient = Math.max(-2, Math.min(2, delta[o] * previous[i]));
        layer.mw[index] = b1 * layer.mw[index] + (1 - b1) * gradient;
        layer.vw[index] = b2 * layer.vw[index] + (1 - b2) * gradient * gradient;
        layer.weights[index] -= lr * (layer.mw[index] / b1Correction) / (Math.sqrt(layer.vw[index] / b2Correction) + eps);
      }
    }
  }
  return loss / OUTPUT_SIZE;
}

function evaluate(count) {
  let mse = 0, decisions = 0, correct = 0;
  for (let n = 0; n < count; n++) {
    const input = scenario(), target = teacher(input), output = forward(input).at(-1);
    for (let i = 0; i < OUTPUT_SIZE; i++) mse += (output[i] - target[i]) ** 2;
    for (const i of [4, 5]) {
      decisions++;
      if ((output[i] > 0) === (target[i] > 0)) correct++;
    }
  }
  return { mse: mse / (count * OUTPUT_SIZE), attackAccuracy: correct / decisions };
}

function cleanTemporaryData() {
  if (fs.existsSync(TEMP_PATH)) fs.rmSync(TEMP_PATH, { recursive: true, force: true });
}

try {
  cleanTemporaryData();
  fs.mkdirSync(TEMP_PATH, { recursive: true });
  let movingLoss = 0;
  for (let step = 1; step <= STEPS; step++) {
    const input = scenario();
    movingLoss = movingLoss * 0.995 + trainOne(input, teacher(input)) * 0.005;
    if (step % 10000 === 0) console.log(`step=${step} movingMse=${movingLoss.toFixed(5)}`);
  }

  const metrics = evaluate(5000);
  const round = value => Number(value.toFixed(7));
  const model = {
    version: 1,
    kind: 'imitation-mlp',
    trainedAt: new Date().toISOString(),
    seed: SEED,
    trainingSteps: STEPS,
    inputSize: INPUT_SIZE,
    outputSize: OUTPUT_SIZE,
    outputs: ['moveX', 'moveY', 'aimX', 'aimY', 'swing', 'throw'],
    validation: { mse: round(metrics.mse), attackAccuracy: round(metrics.attackAccuracy) },
    layers: layers.map(layer => ({
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      weights: Array.from(layer.weights, round),
      biases: Array.from(layer.biases, round),
    })),
  };
  fs.mkdirSync(path.dirname(MODEL_PATH), { recursive: true });
  fs.writeFileSync(MODEL_PATH, `${JSON.stringify(model)}\n`);
  const bytes = fs.statSync(MODEL_PATH).size;
  console.log(`validationMse=${metrics.mse.toFixed(5)} attackAccuracy=${(metrics.attackAccuracy * 100).toFixed(2)}%`);
  console.log(`model=${MODEL_PATH} bytes=${bytes}`);
  if (bytes >= 100 * 1024 * 1024) throw new Error('Permanent model exceeds 100 MB budget');
  if (metrics.mse > 0.22 || metrics.attackAccuracy < 0.88) throw new Error('Training quality gate failed');
} finally {
  cleanTemporaryData();
}
