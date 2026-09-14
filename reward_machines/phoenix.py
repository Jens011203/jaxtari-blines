import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class PhoenixRm(GameRM):
    NUM_FEATURES = 1450
    SCORE_OFFSET, LIVES_OFFSET = -2, -1
    BOSS_ACTIVE_OFFSET = -1302
    BLOCKS_ACTIVE_START, BLOCKS_ACTIVE_END = -650, -488
    ENEMIES_ACTIVE_START, ENEMIES_ACTIVE_END = -1338, -1330
    TOTAL_BLOCKS = 162

    PROP_INDEX = {
        "lost_life": 0,
        "entered_boss": 1,
        "boss_defeated": 2,
        "shield_hit": 3,
        "scored": 4,
    }

    TRANSITIONS = [
        # State 0 (u0): regular wave
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -0.5},
        {"from": 0, "true": ["entered_boss"], "false": ["lost_life"], "to": 1, "reward": 1.0, "option": True},
        {"from": 0, "true": ["scored"], "false": ["lost_life", "entered_boss"], "to": 0, "reward": 0.02},

        # State 1 (u1): boss fight in progress
        {"from": 1, "true": ["lost_life"], "to": 1, "reward": -0.8},
        {"from": 1, "true": ["boss_defeated"], "false": ["lost_life"], "to": 0, "reward": 3.0, "option": True},
        {"from": 1, "true": ["shield_hit"], "false": ["lost_life", "boss_defeated"], "to": 1, "reward": 0.15},
        {"from": 1, "true": ["scored"], "false": ["lost_life", "boss_defeated", "shield_hit"], "to": 1, "reward": 0.05},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 2
    def init_state(self):     return 0
    def terminal_state(self): return -99

    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        score_now = obs[self.SCORE_OFFSET]
        lives_now = obs[self.LIVES_OFFSET]
        boss_active_now = obs[self.BOSS_ACTIVE_OFFSET]
        blocks_active_now = jnp.sum(obs[self.BLOCKS_ACTIVE_START:self.BLOCKS_ACTIVE_END])

        score_prev = obs[self.SCORE_OFFSET - self.NUM_FEATURES]
        lives_prev = obs[self.LIVES_OFFSET - self.NUM_FEATURES]
        boss_active_prev = obs[self.BOSS_ACTIVE_OFFSET - self.NUM_FEATURES]
        blocks_active_prev = jnp.sum(obs[self.BLOCKS_ACTIVE_START - self.NUM_FEATURES:self.BLOCKS_ACTIVE_END - self.NUM_FEATURES])

        lost_life = lives_now < lives_prev
        scored = score_now > score_prev
        entered_boss = (boss_active_now > 0) & (boss_active_prev <= 0)
        boss_defeated = (boss_active_prev > 0) & (boss_active_now <= 0)
        shield_hit = (boss_active_now > 0) & (blocks_active_now < blocks_active_prev)

        return jnp.array([
            lost_life,
            entered_boss,
            boss_defeated,
            shield_hit,
            scored,
        ]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        boss_active = obs[self.BOSS_ACTIVE_OFFSET]
        blocks_active = jnp.sum(obs[self.BLOCKS_ACTIVE_START:self.BLOCKS_ACTIVE_END])
        enemies_active = jnp.sum(obs[self.ENEMIES_ACTIVE_START:self.ENEMIES_ACTIVE_END])

        boss_progress = (self.TOTAL_BLOCKS - blocks_active).astype(jnp.float32)
        wave_progress = (8.0 - enemies_active) * 2.0

        return jnp.where(boss_active > 0, boss_progress, wave_progress)
