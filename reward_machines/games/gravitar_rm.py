import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions, field, field_slice

NUM_FEATURES = 194
SHIP = 0
ENEMIES = 8;                N_ENEMIES = 4
FUEL_TANKS = 40;            N_FUEL_TANKS = 4
SAUCER = 72
UFO = 80
PLANETS = 88;               N_PLANETS = 7
PROJECTILES = 144;          N_PROJECTILES = 4
TERRAIN = 176
REACTOR_DEST = 184
LIVES = 192
FUEL = 193

ACTIVE_SUBOFFSET = 4


class GravitarRm(GameRM):
    """v2 design: adds a small idle/step penalty to discourage the ship
    hovering near the top of the screen instead of engaging with the level."""

    PROP_INDEX = {"lost_life": 0, "fuel_tank_collected": 1, "enemy_destroyed": 2}

    TRANSITIONS = [
        {"from": 0, "true": ["lost_life"], "to": 0, "reward": -1.0},
        {"from": 0, "true": ["fuel_tank_collected"], "false": ["lost_life"], "to": 0, "reward": 0.5},
        {"from": 0, "true": ["enemy_destroyed"], "false": ["lost_life", "fuel_tank_collected"], "to": 0, "reward": 0.3},
        {"from": 0, "false": ["lost_life", "fuel_tank_collected", "enemy_destroyed"], "to": 0, "reward": -0.01},
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

        fuel_active_now  = field_slice(obs, NUM_FEATURES, FUEL_TANKS + ACTIVE_SUBOFFSET * N_FUEL_TANKS, N_FUEL_TANKS, frames_ago=0)
        fuel_active_prev = field_slice(obs, NUM_FEATURES, FUEL_TANKS + ACTIVE_SUBOFFSET * N_FUEL_TANKS, N_FUEL_TANKS, frames_ago=1)

        enemy_active_now  = field_slice(obs, NUM_FEATURES, ENEMIES + ACTIVE_SUBOFFSET * N_ENEMIES, N_ENEMIES, frames_ago=0)
        enemy_active_prev = field_slice(obs, NUM_FEATURES, ENEMIES + ACTIVE_SUBOFFSET * N_ENEMIES, N_ENEMIES, frames_ago=1)

        lost_life           = lives_now < lives_prev
        fuel_tank_collected = jnp.sum(fuel_active_now) < jnp.sum(fuel_active_prev)
        enemy_destroyed     = jnp.sum(enemy_active_now) < jnp.sum(enemy_active_prev)

        return jnp.array([lost_life, fuel_tank_collected, enemy_destroyed]).astype(jnp.int32)
