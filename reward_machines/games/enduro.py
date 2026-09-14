import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class EnduroRm(GameRM):

    NUM_FEATURES = 17
    DIST_LEFT_OFFSET, DIST_RIGHT_OFFSET = -2, -1
    EDGE_THRESHOLD = 0.12
    MAX_CLEARANCE = 0.25

    PROP_INDEX = {
        "near_edge": 0,
    }

    # Single state, two complementary self-loops
    TRANSITIONS = [
        {"from": 0, "true": ["near_edge"], "to": 0, "reward": -0.1},
        {"from": 0, "false": ["near_edge"], "to": 0, "reward": 0.01, "option": True},
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
        dist_left = obs[self.DIST_LEFT_OFFSET]
        dist_right = obs[self.DIST_RIGHT_OFFSET]
        near_edge = (dist_left < self.EDGE_THRESHOLD) | (dist_right < self.EDGE_THRESHOLD)
        return jnp.array([near_edge]).astype(jnp.int32)

    @functools.partial(jax.jit, static_argnums=(0,))
    def potential(self, obs):
        """Phi(s) = normalized clearance to the nearer track edge (higher = safer)."""
        dist_left = obs[self.DIST_LEFT_OFFSET]
        dist_right = obs[self.DIST_RIGHT_OFFSET]
        clearance = jnp.minimum(dist_left, dist_right)
        return jnp.clip(clearance / self.MAX_CLEARANCE, 0.0, 1.0)
