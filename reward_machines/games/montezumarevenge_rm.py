import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field_slice

NUM_FEATURES = 192
ITEMS = 32
N_ITEMS = 3
ACTIVE_SUBOFFSET = 4


class MontezumaRm(GameRM):
    """v2 design: adds a small per-step survival bonus. The agent was dying
    almost immediately in eval; item_collected alone gives no gradient
    toward staying alive longer."""

    PROP_INDEX = {"item_collected": 0}

    TRANSITIONS = [
        {"from": 0, "true": ["item_collected"], "to": 0, "reward": 1.0},
        {"from": 0, "false": ["item_collected"], "to": 0, "reward": 0.01},
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
        items_active_now  = field_slice(obs, NUM_FEATURES, ITEMS + ACTIVE_SUBOFFSET * N_ITEMS, N_ITEMS, frames_ago=0)
        items_active_prev = field_slice(obs, NUM_FEATURES, ITEMS + ACTIVE_SUBOFFSET * N_ITEMS, N_ITEMS, frames_ago=1)

        item_collected = jnp.sum(items_active_now) < jnp.sum(items_active_prev)

        return jnp.array([item_collected]).astype(jnp.int32)
