import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field, field_slice

# Verified via dump_layout.py against the real jaxatari BreakoutObservation
# (player, ball, blocks, lives, score):
NUM_FEATURES = 126
# offsets within one frame
PLAYER = 0          # x,y,width,height,active,visual_id,state,orientation (8)
BALL = 8            # same 8 sub-fields
BLOCKS = 16         # (6, 18) = 108 elements, flattened row-major
LIVES = 124
SCORE = 125
NUM_BLOCKS = 108


class BreakoutRm(GameRM):
    """v1 design. Breakout has essentially no sub-goal structure (a single
    rally loop of hit-brick / lose-ball), so this RM is intentionally a
    single-state shaping machine, similar in spirit to PongRm. Expect this
    to be a weak test case for "does RM help" (useful negative result for
    RQ2) -- iterate reward magnitudes after the first 1M-step runs.
    """

    PROP_INDEX = {"wall_cleared": 0, "brick_hit": 1, "lost_life": 2}

    # Order matters: build_transitions/RewardMachine picks the FIRST matching
    # row, so put the most specific condition first.
    TRANSITIONS = [
        {"from": 0, "true": ["wall_cleared"], "to": 0, "reward": 5.0},
        {"from": 0, "true": ["brick_hit"], "false": ["wall_cleared"], "to": 0, "reward": 0.2},
        {"from": 0, "true": ["lost_life"], "false": ["brick_hit", "wall_cleared"], "to": 0, "reward": -1.0},
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
        lives_now  = field(obs, NUM_FEATURES, LIVES, frames_ago=0)
        lives_prev = field(obs, NUM_FEATURES, LIVES, frames_ago=1)

        blocks_now  = field_slice(obs, NUM_FEATURES, BLOCKS, NUM_BLOCKS, frames_ago=0)
        blocks_prev = field_slice(obs, NUM_FEATURES, BLOCKS, NUM_BLOCKS, frames_ago=1)
        remaining_now  = jnp.sum(blocks_now)
        remaining_prev = jnp.sum(blocks_prev)

        brick_hit    = remaining_now < remaining_prev
        wall_cleared = (remaining_now == 0) & (remaining_prev > 0)
        lost_life    = lives_now < lives_prev

        return jnp.array([wall_cleared, brick_hit, lost_life]).astype(jnp.int32)
