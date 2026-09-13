const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');

const GameMap = require('../src/game/GameMap');
const Types = require('../src/game/Types');
const { buildObservation, INPUT_SIZE } = require('../src/game/ai/NeuralObservation');
const { getNeuralPolicy, MODEL_PATH, MAX_MODEL_BYTES } = require('../src/game/ai/NeuralPolicy');

function fakeBot() {
  return {
    shape: { x: 0, y: 0, radius: 100 }, angle: 0, neuralWanderPhase: 0,
    health: { percent: 0.75 }, inSafezone: false,
    sword: { isAnimationFinished: true, isFlying: false, flyCooldownTime: 0 },
    humans: [], bots: [], mobs: [], projectiles: [], _coins: [], chests: [], ores: [],
    hazards: [], solids: [],
  };
}

test('trained policy is compact, singleton, finite, and bounded', () => {
  assert.ok(fs.statSync(MODEL_PATH).size < MAX_MODEL_BYTES);
  const policy = getNeuralPolicy();
  assert.equal(policy, getNeuralPolicy());
  const output = policy.predict(new Float32Array(INPUT_SIZE));
  assert.equal(output.length, 6);
  for (const value of output) assert.ok(Number.isFinite(value) && value >= -1 && value <= 1);
});

test('observation contains only entities supplied by the visible viewport', () => {
  const bot = fakeBot();
  bot.game = { entities: new Map([[99, { type: Types.Entity.Player, shape: { x: 10, y: 10 } }]]) };
  let observation = buildObservation(bot);
  assert.equal(observation[4], 0, 'global game entities must not leak into observations');

  bot.humans.push({ type: Types.Entity.Player, shape: { x: 1150, y: 0, radius: 100 }, health: { percent: 0.5 } });
  observation = buildObservation(bot);
  assert.equal(observation[4], 1);
  assert.ok(Math.abs(observation[5] - 0.5) < 1e-6);
});

test('server ensures both gauntlet neural players and replaces a removed contender', () => {
  const players = new Set();
  let spawns = 0;
  const map = {
    aiPlayersCount: 10,
    neuralPlayerBot: null,
    neuralPlayerBots: { one: null, two: null },
    game: { players },
    spawnPlayerBot(profile) {
      assert.ok(profile === 'one' || profile === 'two');
      const bot = { isNeuralBot: true, neuralProfile: profile, removed: false, id: ++spawns };
      players.add(bot);
      return bot;
    },
    linkNeuralGauntlet() {
      const { one, two } = this.neuralPlayerBots;
      if (one && two) {
        one.gauntletOpponent = two;
        two.gauntletOpponent = one;
      }
    },
  };

  GameMap.prototype.ensureNeuralPlayerBots.call(map);
  GameMap.prototype.ensureNeuralPlayerBots.call(map);
  assert.equal(spawns, 2);
  assert.equal(map.neuralPlayerBots.one.gauntletOpponent, map.neuralPlayerBots.two);
  map.neuralPlayerBots.two.removed = true;
  GameMap.prototype.ensureNeuralPlayerBots.call(map);
  assert.equal(spawns, 3);
  assert.equal(map.neuralPlayerBot.neuralProfile, 'two');
});
