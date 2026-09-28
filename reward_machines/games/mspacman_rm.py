import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field_slice

NUM_FEATURES = 275
PELLETS = 19
NUM_PELLETS = 252
POWER_PELLETS = 271
NUM_POWER_PELLETS = 4


class MsPacmanRm(GameRM):
    """v2 design: single-state shaping over pellet-eating events, plus a
    small idle/step penalty to discourage aimless back-and-forth movement
    (dithering in corners) that v1 had no signal against."""

    PROP_INDEX = {"level_cleared": 0, "power_pellet_eaten": 1, "pellet_eaten": 2}

    TRANSITIONS = [
        {"from": 0, "true": ["level_cleared"], "to": 0, "reward": 5.0},
        {"from": 0, "true": ["power_pellet_eaten"], "false": ["level_cleared"], "to": 0, "reward": 0.5},
        {"from": 0, "true": ["pellet_eaten"], "false": ["power_pellet_eaten", "level_cleared"], "to": 0, "reward": 0.05},
        {"from": 0, "false": ["level_cleared", "power_pellet_eaten", "pellet_eaten"], "to": 0, "reward": -0.01},
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
        pellets_now  = field_slice(obs, NUM_FEATURES, PELLETS, NUM_PELLETS, frames_ago=0)
        pellets_prev = field_slice(obs, NUM_FEATURES, PELLETS, NUM_PELLETS, frames_ago=1)
        power_now  = field_slice(obs, NUM_FEATURES, POWER_PELLETS, NUM_POWER_PELLETS, frames_ago=0)
        power_prev = field_slice(obs, NUM_FEATURES, POWER_PELLETS, NUM_POWER_PELLETS, frames_ago=1)

        pellets_left_now  = jnp.sum(pellets_now)
        pellets_left_prev = jnp.sum(pellets_prev)

        pellet_eaten       = pellets_left_now < pellets_left_prev
        power_pellet_eaten = jnp.sum(power_now) < jnp.sum(power_prev)
        level_cleared       = (pellets_left_now == 0) & (pellets_left_prev > 0)

        return jnp.array([level_cleared, power_pellet_eaten, pellet_eaten]).astype(jnp.int32)
