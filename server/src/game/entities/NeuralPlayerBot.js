const Player = require('./Player');
const PlayerBot = require('./PlayerBot');
const Types = require('../Types');
const { getNeuralPolicy } = require('../ai/NeuralPolicy');
const { buildObservation } = require('../ai/NeuralObservation');

const DECISION_INTERVAL = 0.1;

class NeuralPlayerBot extends PlayerBot {
  constructor(game, objectData) {
    super(game, objectData);
    this.isNeuralBot = true;
    this.neuralProfile = objectData.neuralProfile || 'one';
    this.skin = this.neuralProfile === 'two' ? 408 : 0;
    this.policy = getNeuralPolicy();
    this.neuralDecisionTimer = 0;
    this.neuralWanderPhase = Math.random() * Math.PI * 2;
    this.neuralAction = new Float32Array(this.policy.outputSize);
  }

  applyInputs(dt) {
    this.inputs.inputUp(Types.Input.SwordSwing);
    this.inputs.inputUp(Types.Input.SwordThrow);
    this.inputs.inputUp(Types.Input.Ability);

    this.neuralWanderPhase += dt * 0.7;
    if (this.neuralWanderPhase > Math.PI * 2) this.neuralWanderPhase -= Math.PI * 2;
    this.neuralDecisionTimer -= dt;

    if (this.neuralDecisionTimer <= 0) {
      this.neuralDecisionTimer = DECISION_INTERVAL;
      // perceive() is the same viewport query used to build a normal player's
      // snapshot. The observation encoder deliberately uses only a subset of it.
      this.perceive();
      this.neuralAction.set(this.policy.predict(buildObservation(this)));
      if (this.neuralProfile === 'two') this.improveCombatAction();
      this.checkUpgrades();
    }

    const baseAngle = this.angle;
    const moveX = this.neuralAction[0];
    const moveY = this.neuralAction[1];
    const aimX = this.neuralAction[2];
    const aimY = this.neuralAction[3];
    const moveMagnitude = Math.min(1, Math.hypot(moveX, moveY));

    if (moveMagnitude > 0.05) {
      this.moveAngle = baseAngle + Math.atan2(moveY, moveX);
      this.mouse = { angle: this.moveAngle, force: moveMagnitude * 150 };
    } else {
      this.mouse = { angle: this.moveAngle, force: 0 };
    }

    if (Math.hypot(aimX, aimY) > 0.05) {
      const desiredAim = baseAngle + Math.atan2(aimY, aimX);
      this.angle = this.turnToward(baseAngle, desiredAim, dt, 7);
    }

    if (!this.inSafezone && this.neuralAction[4] > 0.25) {
      this.inputs.inputDown(Types.Input.SwordSwing);
    }
    if (!this.inSafezone && this.neuralAction[5] > 0.35) {
      this.inputs.inputDown(Types.Input.SwordThrow);
    }

    // This is the normal Player input path: no teleports, injected velocity,
    // boss attacks, or privileged ability activation.
    Player.prototype.applyInputs.call(this, dt);
  }

  improveCombatAction() {
    const o = buildObservation(this);
    if (!o[4]) return;

    const ex = o[5], ey = o[6], distance = o[7];
    const side = Math.sin(this.neuralWanderPhase * 1.7) >= 0 ? 1 : -1;
    if (distance > 0.19) {
      this.neuralAction[0] = ex;
      this.neuralAction[1] = ey;
    } else if (distance < 0.105) {
      this.neuralAction[0] = -ex * 0.7 - ey * side * 0.7;
      this.neuralAction[1] = -ey * 0.7 + ex * side * 0.7;
    } else {
      this.neuralAction[0] = ex * 0.25 - ey * side;
      this.neuralAction[1] = ey * 0.25 + ex * side;
    }
    this.neuralAction[2] = ex;
    this.neuralAction[3] = ey;
    this.neuralAction[4] = o[1] && distance < 0.20 ? 1 : -1;
    this.neuralAction[5] = o[2] && distance >= 0.18 && distance < 0.72 ? 1 : -1;

    // Dodge only projectiles that are close and actually travelling toward us.
    if (o[12] && o[17] < 0.32) {
      const incoming = o[15] * -o[13] + o[16] * -o[14];
      if (incoming > 0.55) {
        const dodgeSide = (-o[16] * -o[13] + o[15] * -o[14]) >= 0 ? 1 : -1;
        this.neuralAction[0] = -o[16] * dodgeSide;
        this.neuralAction[1] = o[15] * dodgeSide;
      }
    }
  }

  damaged(damage, entity, isThrown = false, opts = null) {
    // Skip PlayerBot's goal/flee mutations. Neural contenders react only
    // through their visible observation and normal input policy.
    return Player.prototype.damaged.call(this, damage, entity, isThrown, opts);
  }

  checkUpgrades() {
    if (!this.evolutions || this.evolutions.possibleEvols.size === 0) return;
    const choices = Array.from(this.evolutions.possibleEvols).sort((a, b) => a - b);
    // Use a policy output to choose among legal offers, while guaranteeing that
    // every tier is actually accepted instead of being lost to the base bot's
    // one-shot random roll.
    const signal = Number.isFinite(this.neuralAction[0]) ? this.neuralAction[0] : 0;
    const index = Math.min(choices.length - 1, Math.floor(((signal + 1) / 2) * choices.length));
    this.evolutions.upgrade(choices[Math.max(0, index)]);
  }
}

module.exports = NeuralPlayerBot;
