const Types = require('../Types');

const INPUT_SIZE = 29;
const VIEW_RANGE = 2300;

function clamp(value, min = -1, max = 1) {
  return Math.max(min, Math.min(max, value));
}

function center(entity) {
  if (!entity || !entity.shape) return null;
  if (entity.shape.center && Number.isFinite(entity.shape.center.x)) return entity.shape.center;
  return entity.shape;
}

function radius(entity) {
  if (!entity || !entity.shape) return 0;
  if (Number.isFinite(entity.shape.radius)) return entity.shape.radius;
  const b = entity.shape.boundary;
  return b ? Math.max(b.width, b.height) / 2 : 0;
}

function nearest(bot, groups) {
  let result = null;
  let distanceSq = Infinity;
  for (const group of groups) {
    for (const entity of group || []) {
      const p = center(entity);
      if (!p || entity.removed || entity === bot) continue;
      const dx = p.x - bot.shape.x;
      const dy = p.y - bot.shape.y;
      const d2 = dx * dx + dy * dy;
      if (d2 < distanceSq) {
        distanceSq = d2;
        result = entity;
      }
    }
  }
  return result ? { entity: result, distance: Math.sqrt(distanceSq) } : null;
}

function localVector(bot, entity, scale = VIEW_RANGE) {
  const p = center(entity);
  const dx = p.x - bot.shape.x;
  const dy = p.y - bot.shape.y;
  const cos = Math.cos(bot.angle);
  const sin = Math.sin(bot.angle);
  return [clamp((dx * cos + dy * sin) / scale), clamp((-dx * sin + dy * cos) / scale)];
}

function localDirection(bot, angle) {
  const relative = (Number.isFinite(angle) ? angle : 0) - bot.angle;
  return [Math.cos(relative), Math.sin(relative)];
}

function buildObservation(bot) {
  const input = new Float32Array(INPUT_SIZE);
  input[0] = clamp(bot.health ? bot.health.percent : 1, 0, 1);
  input[1] = bot.sword && bot.sword.isAnimationFinished && !bot.sword.isFlying ? 1 : 0;
  input[2] = bot.sword && !bot.sword.isFlying && bot.sword.flyCooldownTime <= 0 ? 1 : 0;
  input[3] = bot.inSafezone ? 1 : 0;

  const opponent = bot.gauntletOpponent;
  const opponentVisible = opponent && !opponent.removed
    && Array.isArray(bot.viewportEntityIds) && bot.viewportEntityIds.includes(opponent.id);
  const enemy = opponent
    ? (opponentVisible
      ? { entity: opponent, distance: Math.hypot(opponent.shape.x - bot.shape.x, opponent.shape.y - bot.shape.y) }
      : null)
    : nearest(bot, [bot.humans, bot.bots, bot.mobs]);
  if (enemy) {
    const [x, y] = localVector(bot, enemy.entity);
    input[4] = 1; input[5] = x; input[6] = y;
    input[7] = clamp(enemy.distance / VIEW_RANGE, 0, 1);
    input[8] = clamp(enemy.entity.health ? enemy.entity.health.percent : 1, 0, 1);
    input[9] = enemy.entity.type === Types.Entity.Player ? 1 : 0;
    input[10] = Types.Groups.Mobs.includes(enemy.entity.type) ? 1 : 0;
    input[11] = clamp(radius(enemy.entity) / 500, 0, 1);
  }

  const projectile = nearest(bot, [bot.projectiles]);
  if (projectile) {
    const [x, y] = localVector(bot, projectile.entity);
    const [dx, dy] = localDirection(bot, projectile.entity.angle);
    input[12] = 1; input[13] = x; input[14] = y;
    input[15] = dx; input[16] = dy;
    input[17] = clamp(projectile.distance / VIEW_RANGE, 0, 1);
  }

  const resource = nearest(bot, [bot._coins, bot.chests, bot.ores]);
  if (resource) {
    const [x, y] = localVector(bot, resource.entity);
    input[18] = 1; input[19] = x; input[20] = y;
    input[21] = clamp(resource.distance / VIEW_RANGE, 0, 1);
    input[22] = resource.entity.type === Types.Entity.Coin ? 1 : -1;
  }

  const obstacle = nearest(bot, [bot.hazards, bot.solids]);
  if (obstacle) {
    const [x, y] = localVector(bot, obstacle.entity);
    input[23] = 1; input[24] = x; input[25] = y;
    input[26] = clamp(obstacle.distance / VIEW_RANGE, 0, 1);
  }

  input[27] = Math.sin(bot.neuralWanderPhase || 0);
  input[28] = Math.cos(bot.neuralWanderPhase || 0);
  return input;
}

module.exports = { buildObservation, INPUT_SIZE, VIEW_RANGE };
