import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class AsteroidsRm(GameRM):

    NUM_FEATURES = 162
    SCORE_OFFSET, LIVES_OFFSET = -2, -1

    PROP_INDEX = {
        "lost_life": 0,
        "gained_life": 1,
        "scored": 2,
    }

    TRANSITIONS = [
        # State 0 (u0): no bonus life earned yet since the last death
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -0.5},
        {"from": 0, "true": ["gained_life"], "false": ["lost_life"], "to": 1, "reward": 1.0, "option": True},
        {"from": 0, "true": ["scored"], "false": ["lost_life", "gained_life"], "to": 0, "reward": 0.02},

        # State 1 (u1): at least one bonus life earned since the last death
        {"from": 1, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 1, "true": ["gained_life"], "false": ["lost_life"], "to": 1, "reward": 0.5},
        {"from": 1, "true": ["scored"], "false": ["lost_life", "gained_life"], "to": 1, "reward": 0.05},
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
        score_prev = obs[self.SCORE_OFFSET - self.NUM_FEATURES]
        lives_prev = obs[self.LIVES_OFFSET - self.NUM_FEATURES]

        lost_life = lives_now < lives_prev
        gained_life = lives_now > lives_prev
        scored = score_now > score_prev

        return jnp.array([lost_life, gained_life, scored]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Phi(s) = current score, scaled down."""
        score = obs[self.SCORE_OFFSET]
        return jnp.clip(score / 1000.0, 0.0, 100.0)
