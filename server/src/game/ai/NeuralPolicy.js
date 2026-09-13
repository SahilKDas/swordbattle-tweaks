const fs = require('fs');
const path = require('path');

const MODEL_PATH = path.join(__dirname, 'models', 'neural-player-v1.json');
const MAX_MODEL_BYTES = 100 * 1024 * 1024;

function assertLayer(layer, expectedInputs, name) {
  if (!layer || !Number.isInteger(layer.inputSize) || !Number.isInteger(layer.outputSize)) {
    throw new Error(`Invalid neural policy layer: ${name}`);
  }
  if (layer.inputSize !== expectedInputs
    || layer.weights.length !== layer.inputSize * layer.outputSize
    || layer.biases.length !== layer.outputSize) {
    throw new Error(`Invalid neural policy dimensions: ${name}`);
  }
  if (![...layer.weights, ...layer.biases].every(Number.isFinite)) {
    throw new Error(`Non-finite neural policy weight: ${name}`);
  }
}

class NeuralPolicy {
  constructor(modelPath = MODEL_PATH) {
    const stat = fs.statSync(modelPath);
    if (stat.size > MAX_MODEL_BYTES) {
      throw new Error(`Neural policy exceeds ${MAX_MODEL_BYTES} bytes`);
    }
    const model = JSON.parse(fs.readFileSync(modelPath, 'utf8'));
    if (!Array.isArray(model.layers) || model.layers.length < 2) {
      throw new Error('Neural policy must contain at least two layers');
    }

    let expected = model.inputSize;
    for (let i = 0; i < model.layers.length; i++) {
      assertLayer(model.layers[i], expected, `layer ${i}`);
      expected = model.layers[i].outputSize;
    }
    if (expected !== model.outputSize) throw new Error('Neural policy output size mismatch');

    this.version = model.version;
    this.inputSize = model.inputSize;
    this.outputSize = model.outputSize;
    this.layers = model.layers.map(layer => ({
      inputSize: layer.inputSize,
      outputSize: layer.outputSize,
      weights: Float32Array.from(layer.weights),
      biases: Float32Array.from(layer.biases),
      output: new Float32Array(layer.outputSize),
    }));
  }

  predict(input) {
    if (!input || input.length !== this.inputSize) {
      throw new Error(`Expected ${this.inputSize} neural inputs`);
    }
    let source = input;
    for (const layer of this.layers) {
      const { inputSize, outputSize, weights, biases, output } = layer;
      for (let o = 0; o < outputSize; o++) {
        let sum = biases[o];
        const row = o * inputSize;
        for (let i = 0; i < inputSize; i++) sum += weights[row + i] * source[i];
        output[o] = Math.tanh(sum);
      }
      source = output;
    }
    return source;
  }
}

let singleton = null;

function getNeuralPolicy() {
  if (!singleton) singleton = new NeuralPolicy();
  return singleton;
}

module.exports = { NeuralPolicy, getNeuralPolicy, MODEL_PATH, MAX_MODEL_BYTES };
