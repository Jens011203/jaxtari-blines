import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class AsteroidsRm(GameRM):

    NUM_FEATURES = 162
    SCORE_OFFSET, LIVES_OFFSET = -2, -1
    WIDTH_START, WIDTH_END = -104, -87
    MAX_MASS = 0.6

    PROP_INDEX = {
        "lost_life": 0,
        "wave_cleared": 1,
        "scored": 2,
    }

    TRANSITIONS = [
        # Single state u0: everything is a self-loop. No information gained by using multiple states
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -0.5},
        {"from": 0, "true": ["wave_cleared"], "false": ["lost_life"], "to": 0, "reward": 1.0, "option": True},
        {"from": 0, "true": ["scored"], "false": ["lost_life", "wave_cleared"], "to": 0, "reward": 0.02},
    ]

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return 1
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
        score_prev = obs[self.SCORE_OFFSET - self.NUM_FEATURES]
        lives_prev = obs[self.LIVES_OFFSET - self.NUM_FEATURES]

        # Field "mass" = sum of normalized asteroid widths
        mass_now = jnp.sum(obs[self.WIDTH_START:self.WIDTH_END])
        mass_prev = jnp.sum(
            obs[self.WIDTH_START - self.NUM_FEATURES:self.WIDTH_END - self.NUM_FEATURES]
        )

        lost_life = lives_now < lives_prev
        scored = score_now > score_prev
        wave_cleared = mass_now > mass_prev + 1e-4

        return jnp.array([lost_life, wave_cleared, scored]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        mass = jnp.sum(obs[self.WIDTH_START:self.WIDTH_END])
        return jnp.clip(1.0 - mass / self.MAX_MASS, 0.0, 1.0)
