const test = require('node:test');
const assert = require('node:assert/strict');
const Spectator = require('../src/game/Spectator');

test('bot viewer spectator follows the live neural player exactly', () => {
  const neuralPlayerBot = { removed: false, shape: { x: 1234, y: -567 } };
  const game = {
    map: { neuralPlayerBot, safezone: null, shape: { randomSpawnInside() {} } },
    entitiesQuadtree: { get: () => [] },
  };
  const spectator = new Spectator(game, { player: null });
  spectator.isSpectating = true;
  spectator.followNeuralPlayer = true;
  spectator.update(50);
  assert.equal(spectator.shape.x, 1234);
  assert.equal(spectator.shape.y, -567);
});
