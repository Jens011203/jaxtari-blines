"""
Reward machine for JAXAtari Tennis.

The Tennis-specific reward machine uses reliable events observable from the
stacked object-centric observation. Noisy racket-return events are deliberately
excluded from v0 because validation against the internal `last_hit` state showed
that they cannot be detected reliably under frame skip 4.
"""

import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class TennisRm(GameRM):
    # Proposition order must match get_events().
    PROP_INDEX = {
        "rally_active": 0,
        "player_point": 1,
        "enemy_point": 2,
        "player_game_progress": 3,
        "enemy_game_progress": 4,
    }

    # Two phases:
    # u0 = between points / serve phase
    # u1 = active rally
    #
    # Transition order defines priority.
    # Higher-level game progress must be checked before ordinary point events,
    # because both may be true on a game-winning point.
    TRANSITIONS = [
        # ---- u0: POINT_START / SERVE ---------------------------------------
        {"from": 0, "true": ["player_game_progress"],
         "to": 0, "reward": 3.0},

        {"from": 0, "true": ["enemy_game_progress"],
         "to": 0, "reward": -3.0},

        {"from": 0, "true": ["player_point"],
         "to": 0, "reward": 1.0},

        {"from": 0, "true": ["enemy_point"],
         "to": 0, "reward": -1.0},

        {"from": 0, "true": ["rally_active"],
         "to": 1, "reward": 0.25},

        # ---- u1: RALLY ------------------------------------------------------
        {"from": 1, "true": ["player_game_progress"],
         "to": 0, "reward": 3.0},

        {"from": 1, "true": ["enemy_game_progress"],
         "to": 0, "reward": -3.0},

        {"from": 1, "true": ["player_point"],
         "to": 0, "reward": 1.0},

        {"from": 1, "true": ["enemy_point"],
         "to": 0, "reward": -1.0},
    ]

    def __init__(self):
        (
            self._from,
            self._rt,
            self._rf,
            self._to,
            self._rew,
        ) = build_transitions(
            len(self.PROP_INDEX),
            self.PROP_INDEX,
            self.TRANSITIONS,
        )

    def num_states(self):
        return 2

    def init_state(self):
        return 0

    def terminal_state(self):
        return -99

    def from_states(self):
        return self._from

    def require_true(self):
        return self._rt

    def require_false(self):
        return self._rf

    def to_states(self):
        return self._to

    def rewards(self):
        return self._rew

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        # One Tennis object-centric frame contains 29 features.
        # The full Double-DQN input contains four stacked frames.
        NUM_FEATURES = 29

        IS_SERVING = 24
        PLAYER_POINTS = 25
        ENEMY_POINTS = 26
        PLAYER_GAME_SCORE = 27
        ENEMY_GAME_SCORE = 28

        # Newest frame
        serve_now = obs[-NUM_FEATURES + IS_SERVING]

        player_points_now = obs[-NUM_FEATURES + PLAYER_POINTS]
        enemy_points_now = obs[-NUM_FEATURES + ENEMY_POINTS]

        player_game_now = obs[-NUM_FEATURES + PLAYER_GAME_SCORE]
        enemy_game_now = obs[-NUM_FEATURES + ENEMY_GAME_SCORE]

        # Previous observed frame
        player_points_prev = obs[-2 * NUM_FEATURES + PLAYER_POINTS]
        enemy_points_prev = obs[-2 * NUM_FEATURES + ENEMY_POINTS]

        player_game_prev = obs[-2 * NUM_FEATURES + PLAYER_GAME_SCORE]
        enemy_game_prev = obs[-2 * NUM_FEATURES + ENEMY_GAME_SCORE]

        # The rally is active whenever the environment is no longer in its
        # serving phase. This also handles AtariWrapper(first_fire=True),
        # where reset may already return an active rally.
        rally_active = serve_now < 0.5

        player_game_progress = player_game_now > player_game_prev
        enemy_game_progress = enemy_game_now > enemy_game_prev

        # On the game-winning point the ordinary point counter resets to zero.
        # Therefore higher-level progress must also count as a scored point.
        player_point = (
            (player_points_now > player_points_prev)
            | player_game_progress
        )

        enemy_point = (
            (enemy_points_now > enemy_points_prev)
            | enemy_game_progress
        )

        return jnp.array([
            rally_active,
            player_point,
            enemy_point,
            player_game_progress,
            enemy_game_progress,
        ]).astype(jnp.int32)