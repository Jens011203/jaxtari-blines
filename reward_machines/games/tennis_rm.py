"""
Reward machine for JAXAtari Tennis.

The Tennis RM uses events that can be derived reliably from the stacked
object-centric observation.

Racket-return events are deliberately excluded because validation against the
internal `last_hit` state showed that they cannot be detected reliably under
frame skip 4.

The rally phase is detected from ball motion rather than from `is_serving`.
Validation showed that `is_serving == False` is not sufficient to identify an
active rally, because it can already be false during the post-point pause.

The current RM therefore models three temporal phases:

    u0 = READY
         A stopped/waiting phase before the next rally.

    u1 = RALLY
         The ball is actively in play.

    u2 = POST_POINT
         A point has just been scored. The RM waits until a stopped-ball
         observation has been seen before allowing a new rally.

This prevents stale motion in the stacked observation immediately after a
point from being mistaken for the start of the next rally.
"""

import functools

import jax
import jax.numpy as jnp

from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions


class TennisRm(GameRM):
    # Proposition order must match get_events().
    PROP_INDEX = {
        "ball_in_play": 0,
        "player_point": 1,
        "enemy_point": 2,
        "player_game_progress": 3,
        "enemy_game_progress": 4,
    }

    # RM states:
    #
    # u0 = READY
    # u1 = RALLY
    # u2 = POST_POINT
    #
    # Higher-level game progress transitions come before point transitions
    # because both may be true on a game-winning point.
    TRANSITIONS = [
        # ---- u0: READY ------------------------------------------------------
        {"from": 0, "true": ["player_game_progress"],
         "to": 2, "reward": 3.0},

        {"from": 0, "true": ["enemy_game_progress"],
         "to": 2, "reward": -3.0},

        {"from": 0, "true": ["player_point"],
         "to": 2, "reward": 1.0},

        {"from": 0, "true": ["enemy_point"],
         "to": 2, "reward": -1.0},

        {"from": 0, "true": ["ball_in_play"],
         "to": 1, "reward": 0.25},

        # ---- u1: ACTIVE RALLY ----------------------------------------------
        {"from": 1, "true": ["player_game_progress"],
         "to": 2, "reward": 3.0},

        {"from": 1, "true": ["enemy_game_progress"],
         "to": 2, "reward": -3.0},

        {"from": 1, "true": ["player_point"],
         "to": 2, "reward": 1.0},

        {"from": 1, "true": ["enemy_point"],
         "to": 2, "reward": -1.0},

        # ---- u2: POST-POINT -------------------------------------------------
        # Wait until stale movement from the previous rally has disappeared.
        {"from": 2, "false": ["ball_in_play"],
         "to": 0, "reward": 0.0},
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
        return 3

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
        NUM_FEATURES = 29

        BALL_X = 16
        BALL_Y = 17

        PLAYER_POINTS = 25
        ENEMY_POINTS = 26
        PLAYER_GAME_SCORE = 27
        ENEMY_GAME_SCORE = 28

        # --------------------------------------------------------------
        # Ball-in-play detector
        # --------------------------------------------------------------
        EPS = 1e-6

        ball_x_f1 = obs[-3 * NUM_FEATURES + BALL_X]
        ball_y_f1 = obs[-3 * NUM_FEATURES + BALL_Y]

        ball_x_f2 = obs[-2 * NUM_FEATURES + BALL_X]
        ball_y_f2 = obs[-2 * NUM_FEATURES + BALL_Y]

        ball_x_f3 = obs[-NUM_FEATURES + BALL_X]
        ball_y_f3 = obs[-NUM_FEATURES + BALL_Y]

        m2 = (
            (jnp.abs(ball_x_f2 - ball_x_f1) > EPS)
            | (jnp.abs(ball_y_f2 - ball_y_f1) > EPS)
        )

        m3 = (
            (jnp.abs(ball_x_f3 - ball_x_f2) > EPS)
            | (jnp.abs(ball_y_f3 - ball_y_f2) > EPS)
        )

        ball_in_play = m2 & m3

        # --------------------------------------------------------------
        # Score events
        # --------------------------------------------------------------
        player_points_now = obs[-NUM_FEATURES + PLAYER_POINTS]
        enemy_points_now = obs[-NUM_FEATURES + ENEMY_POINTS]

        player_game_now = obs[-NUM_FEATURES + PLAYER_GAME_SCORE]
        enemy_game_now = obs[-NUM_FEATURES + ENEMY_GAME_SCORE]

        player_points_prev = obs[-2 * NUM_FEATURES + PLAYER_POINTS]
        enemy_points_prev = obs[-2 * NUM_FEATURES + ENEMY_POINTS]

        player_game_prev = obs[-2 * NUM_FEATURES + PLAYER_GAME_SCORE]
        enemy_game_prev = obs[-2 * NUM_FEATURES + ENEMY_GAME_SCORE]

        player_game_progress = player_game_now > player_game_prev
        enemy_game_progress = enemy_game_now > enemy_game_prev

        player_point = (
            (player_points_now > player_points_prev)
            | player_game_progress
        )

        enemy_point = (
            (enemy_points_now > enemy_points_prev)
            | enemy_game_progress
        )

        return jnp.array([
            ball_in_play,
            player_point,
            enemy_point,
            player_game_progress,
            enemy_game_progress,
        ]).astype(jnp.int32)