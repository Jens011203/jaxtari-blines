import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field

# Verified via reward_machines/tools/dump_layout.py against the real jaxatari
# SkiingObservation (skier, flags[2], trees[4], moguls[2], successful_gates).
NUM_FEATURES = 73
SUCCESSFUL_GATES = 72

# --------------------------------------------------------------------------
# BUG FIX (found after the v1 test run showed 0 fires in 2000 random steps
# across 8 envs, which is implausible for a common event -- gates come by
# frequently even under random actions).
#
# Traced it in jaxatari's own src/jaxatari/games/jax_skiing.py:
#   - state reset:  successful_gates = 20                          (L410)
#   - on each step: new_score = state.successful_gates - gates_scored,
#                    gates_scored = jnp.sum(gate_pass)               (L764-765)
#   - end-of-episode: missed_gates = 20 - state.successful_gates     (L882)
#   - the built-in env reward itself is defined as
#     (previous_state.successful_gates - state.successful_gates)     (L889)
#
# So despite its name, `successful_gates` is actually a COUNTDOWN of gates
# REMAINING (starts at 20, decreases by 1 each time a gate is passed), not a
# count of gates passed. v1 had this backwards (`gates_now > gates_prev`).
# Fixed below to `gates_now < gates_prev`. If you add a game to this list
# later, don't trust a field's name -- verify direction of travel like this,
# either from the source's own step()/reward function or empirically with
# reward_machines/tools/dump_layout.py plus a short scripted rollout.
# --------------------------------------------------------------------------

# Skiing has no `lives` field (a run is a single downhill slide, ALE scores
# it by time + a penalty per missed gate), and `successful_gates` is already
# a ready-made progress counter -- this is the best-behaved game of the five
# for RM design, closest in spirit to Icarte et al.'s sub-goal chain example.
class SkiingRm(GameRM):
    """v1 design (fixed): +1 reward each time the remaining-gates counter
    DECREASES (i.e. a gate is passed). Collision-with-tree/mogul penalties
    are left out of v1 (no clearly documented "collision" flag was found in
    skier.state/active during this pass -- inspect a rollout with the test
    suite before adding one)."""

    PROP_INDEX = {"gate_passed": 0}

    TRANSITIONS = [
        {"from": 0, "true": ["gate_passed"], "to": 0, "reward": 1.0},
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
        gates_now  = field(obs, NUM_FEATURES, SUCCESSFUL_GATES, frames_ago=0)
        gates_prev = field(obs, NUM_FEATURES, SUCCESSFUL_GATES, frames_ago=1)

        gate_passed = gates_now < gates_prev  # counter counts DOWN, see note above

        return jnp.array([gate_passed]).astype(jnp.int32)
