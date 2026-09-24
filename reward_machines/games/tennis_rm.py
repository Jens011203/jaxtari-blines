"""
Reward machine for JAXAtari Tennis.

Tennis RM v3 extends the validated temporal phase machine from v2 with
an intermediate serve-preparation subgoal.

Validation motivating v3:
- v2 correctly detected READY / RALLY / POST_POINT phases.
- A trained DDQN+CRM policy nevertheless became stuck during player serve.
- A causal RIGHTFIRE intervention showed that the agent first has to move
  into hitting range before FIRE can start the rally.
- Observation-space validation showed that player-ball serving alignment
  can be detected reliably using both horizontal distance and the vertical
  hit-line condition.
- X distance alone is insufficient because it also fires during enemy serve.

RM states:

    u0 = READY
         Waiting for the next rally.

    u1 = SERVE_READY
         The player has reached the serve-ball hitting region at least once
         during the current pre-rally phase.

    u2 = RALLY
         The ball is actively in play.

    u3 = POST_POINT
         A point has just been scored. Wait until stale stacked ball motion
         has disappeared before returning to READY.

The old v2 rally-start reward (+0.25) is not increased.
For a player serve it is split into:

    READY -> SERVE_READY : +0.10
    SERVE_READY -> RALLY : +0.15

For rallies that start without the player-specific serve-preparation event
(e.g. enemy serve), READY -> RALLY still receives +0.25.
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
        "serve_ready": 1,
        "player_point": 2,
        "enemy_point": 3,
        "player_game_progress": 4,
        "enemy_game_progress": 5,
    }

    # RM states:
    #
    # u0 = READY
    # u1 = SERVE_READY
    # u2 = RALLY
    # u3 = POST_POINT
    #
    # Higher-level progress transitions come before ordinary point
    # transitions because both may be true on a game-winning point.
    TRANSITIONS = [
        # ---- u0: READY -------------------------------------------------
        {"from": 0, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 0, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 0, "true": ["player_point"],
         "to": 3, "reward": 1.0},

        {"from": 0, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # If a rally starts without a player serve-preparation phase
        # (for example enemy serve), preserve the original v2 reward.
        {"from": 0, "true": ["ball_in_play"],
         "to": 2, "reward": 0.25},

        # Player reached a valid serve hitting region.
        {"from": 0, "true": ["serve_ready"],
         "to": 1, "reward": 0.10},

        # ---- u1: SERVE_READY -------------------------------------------
        {"from": 1, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 1, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 1, "true": ["player_point"],
         "to": 3, "reward": 1.0},

        {"from": 1, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # Second half of the old +0.25 rally-start reward.
        {"from": 1, "true": ["ball_in_play"],
         "to": 2, "reward": 0.15},

        # IMPORTANT:
        # There is deliberately no SERVE_READY -> READY transition when
        # serve_ready becomes false. Otherwise the agent could repeatedly
        # move in/out of alignment and farm the +0.10 reward.

        # ---- u2: RALLY -------------------------------------------------
        {"from": 2, "true": ["player_game_progress"],
         "to": 3, "reward": 3.0},

        {"from": 2, "true": ["enemy_game_progress"],
         "to": 3, "reward": -3.0},

        {"from": 2, "true": ["player_point"],
         "to": 3, "reward": 1.0},

        {"from": 2, "true": ["enemy_point"],
         "to": 3, "reward": -1.0},

        # ---- u3: POST_POINT --------------------------------------------
        # Wait until stale movement from the previous rally disappears.
        {"from": 3, "false": ["ball_in_play"],
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
        return 4

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

        # ObjectObservation layout:
        # x, y, width, height, active, state, vis_id, orientation
        PLAYER_X = 0
        PLAYER_Y = 1

        BALL_X = 16
        BALL_Y = 17

        SERVE = 24

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
        # Serve-preparation detector
        # --------------------------------------------------------------
        #
        # Empirical validation against the raw Tennis hit geometry:
        #
        # |player_x - ball_x| <= 0.08
        #
        # perfectly separated player-serving hit-ready vs non-ready
        # samples in the validation rollout.
        #
        # X alone is NOT sufficient because enemy serves have a similar
        # horizontal arrangement. Therefore we additionally require the
        # player's vertical hit line to coincide with the ball.
        #
        # Tennis constants:
        # FRAME_HEIGHT  = 210
        # PLAYER_HEIGHT = 23
        # hit tolerance = 3 pixels

        player_x_now = obs[-NUM_FEATURES + PLAYER_X]
        player_y_now = obs[-NUM_FEATURES + PLAYER_Y]

        ball_x_now = obs[-NUM_FEATURES + BALL_X]
        ball_y_now = obs[-NUM_FEATURES + BALL_Y]

        serve_now = obs[-NUM_FEATURES + SERVE] > 0.5

        X_THRESHOLD = 0.08
        PLAYER_HEIGHT_NORMALIZED = 23.0 / 210.0
        Y_THRESHOLD = 3.0 / 210.0

        x_aligned = (
            jnp.abs(player_x_now - ball_x_now)
            <= X_THRESHOLD
        )

        y_aligned = (
            jnp.abs(
                (
                    player_y_now
                    + PLAYER_HEIGHT_NORMALIZED
                )
                - ball_y_now
            )
            <= Y_THRESHOLD
        )

        serve_ready = (
            serve_now
            & x_aligned
            & y_aligned
        )

        # --------------------------------------------------------------
        # Score events
        # --------------------------------------------------------------
        player_points_now = obs[
            -NUM_FEATURES + PLAYER_POINTS
        ]
        enemy_points_now = obs[
            -NUM_FEATURES + ENEMY_POINTS
        ]

        player_game_now = obs[
            -NUM_FEATURES + PLAYER_GAME_SCORE
        ]
        enemy_game_now = obs[
            -NUM_FEATURES + ENEMY_GAME_SCORE
        ]

        player_points_prev = obs[
            -2 * NUM_FEATURES + PLAYER_POINTS
        ]
        enemy_points_prev = obs[
            -2 * NUM_FEATURES + ENEMY_POINTS
        ]

        player_game_prev = obs[
            -2 * NUM_FEATURES + PLAYER_GAME_SCORE
        ]
        enemy_game_prev = obs[
            -2 * NUM_FEATURES + ENEMY_GAME_SCORE
        ]

        player_game_progress = (
            player_game_now > player_game_prev
        )

        enemy_game_progress = (
            enemy_game_now > enemy_game_prev
        )

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
            serve_ready,
            player_point,
            enemy_point,
            player_game_progress,
            enemy_game_progress,
        ]).astype(jnp.int32)