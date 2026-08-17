import functools
import jax
import jax.numpy as jnp
from reward_machines.games.game_rm import GameRM
from reward_machines.games.utils import build_transitions

# ---------------------------------------------------------------------------
# Reward / state design
# ---------------------------------------------------------------------------
# The RM state encodes (deaths so far, the floor the kangaroo is standing on):
#
#       state = 4 * deaths + floor        deaths in 0..2, floor in 0..3
#
# so 0..3 = full lives, 4..7 = one life lost, 8..11 = two lost, then terminal.
# The state always matches the real position -- floor 2 means the kangaroo is on
# floor 2 right now, not that it was there once.
#
# `lives` is not in the observation, but it starts at 3, never increases (there
# are no bonus lives) and every death is detectable -- counting deaths in the RM
# state therefore reconstructs it exactly.
#
# Floor rewards are pure shaping: r = PHI[to] - PHI[from], so every closed path
# sums to exactly zero. Riding a ladder up and down pays nothing and no
# anti-farming special case is needed.
#
# The goal bonus sits on `level_changed`, the observed completion, not on
# reaching the top platform. Standing on floor 3 does imply the level will
# advance 256 ticks later, but that coupling lives inside the engine and is not
# something the observation confirms -- so the payout waits for the platform
# layout to actually change. Every transition that resets the floor subtracts
# PHI[floor], which makes the return for a completed level exactly LEVEL_REWARD,
# independent of the path taken through the floors.
# ---------------------------------------------------------------------------

PHI = [0.0, 1.0, 2.0, 3.0]  # potential per floor, cancels over any closed path
LEVEL_REWARD = 10.0         # child rescued, verified by the level change
FRUIT_REWARD = 0.5
MONKEY_REWARD = 0.3
DEATH_PENALTY = 1.0          # on top of the potential that is lost
GAME_OVER_PENALTY = 3.0      # extra cost for spending the last life

NUM_FLOORS = 4
MAX_DEATHS = 3


def _build_table():
    """Generate the transition table for every (deaths, floor) combination."""
    rows = []
    for deaths in range(MAX_DEATHS):
        for floor in range(NUM_FLOORS):
            src = NUM_FLOORS * deaths + floor
            last_life = deaths == MAX_DEATHS - 1

            # 1) died -- lose the accumulated potential plus a flat penalty
            penalty = DEATH_PENALTY + PHI[floor] + (GAME_OVER_PENALTY if last_life else 0.0)
            rows.append({
                "from": src,
                "true": ["died"],
                "to": -99 if last_life else NUM_FLOORS * (deaths + 1),
                "reward": -penalty,
            })
            blocked = ["died"]

            # 2) level_changed -- the child is rescued and a new level teleports
            #    the player to the ground floor. Ranked above the floor props so
            #    the teleport is not mistaken for a fall.
            rows.append({
                "from": src,
                "true": ["level_changed"],
                "false": list(blocked),
                "to": NUM_FLOORS * deaths,
                "reward": LEVEL_REWARD - PHI[floor],
            })
            blocked.append("level_changed")

            # 3) next_floor_reached / fell_down -- potential difference, so the
            #    two directions cancel out exactly
            for target in range(NUM_FLOORS - 1, -1, -1):
                if target == floor:
                    continue  # standing still fires no transition
                rows.append({
                    "from": src,
                    "true": [f"on_floor_{target}"],
                    "false": list(blocked),
                    "to": NUM_FLOORS * deaths + target,
                    "reward": PHI[target] - PHI[floor],
                })
                blocked.append(f"on_floor_{target}")

            # 4) fruit_collected / monkey_punched -- stay put, collect points
            rows.append({
                "from": src, "true": ["fruit_collected"], "false": list(blocked),
                "to": src, "reward": FRUIT_REWARD,
            })
            blocked.append("fruit_collected")
            rows.append({
                "from": src, "true": ["monkey_punched"], "false": list(blocked),
                "to": src, "reward": MONKEY_REWARD,
            })
    return rows


class KangarooRm(GameRM):
    """Reward machine for JAXAtari Kangaroo.

    Rewarded events: completing a level, reaching another floor (up or down),
    collecting fruit, punching a monkey, dying.
    """

    PROP_INDEX = {
        "died": 0,
        "level_changed": 1,
        "on_floor_3": 2,   # top platform -- child reached, level completes
        "on_floor_2": 3,
        "on_floor_1": 4,
        "on_floor_0": 5,
        "fruit_collected": 6,
        "monkey_punched": 7,
    }

    # Priority: died > level_changed > floor change > fruit > monkey
    TRANSITIONS = _build_table()

    # ---- Flat layout of KangarooObservation (ravel_pytree order) ----
    NUM_FEATURES = 440
    PLAYER_Y, PLAYER_H = 1, 3
    PLATFORM_Y = slice(28, 48)
    PLATFORM_ACTIVE = slice(88, 108)
    FRUIT_ACTIVE = slice(340, 343)
    MONKEY_STATE = slice(400, 404)

    # NormalizeObservationWrapper bounds; set both to 1.0 if you skip that wrapper.
    Y_SCALE, ID_SCALE = 210.0, 255.0

    # Main platforms sit at y = 172 / 124 / 76 / 28 in every level.
    FLOOR_EDGES = jnp.array([28.5, 76.5, 124.5])
    # While crashing the player sinks until y + PLAYER_HEIGHT > SCREEN_HEIGHT.
    CRASH_Y = 186.0

    def __init__(self):
        (self._from, self._rt, self._rf, self._to, self._rew) = build_transitions(
            len(self.PROP_INDEX), self.PROP_INDEX, self.TRANSITIONS
        )

    def num_states(self):     return NUM_FLOORS * MAX_DEATHS
    def init_state(self):     return 0
    def terminal_state(self): return -99

    def from_states(self):    return self._from
    def require_true(self):   return self._rt
    def require_false(self):  return self._rf
    def to_states(self):      return self._to
    def rewards(self):        return self._rew

    @staticmethod
    def decode(rm_state):
        """Split an RM state back into (current_floor, lives_left)."""
        return rm_state % NUM_FLOORS, MAX_DEATHS - rm_state // NUM_FLOORS

    @functools.partial(jax.jit, static_argnums=(0,))
    def get_events(self, obs):
        frames = obs.reshape(-1, self.NUM_FEATURES)
        now, prev = frames[-1], frames[-2]

        player_y = now[self.PLAYER_Y] * self.Y_SCALE
        player_y_prev = prev[self.PLAYER_Y] * self.Y_SCALE
        feet = player_y + now[self.PLAYER_H] * self.Y_SCALE
        feet_prev = player_y_prev + prev[self.PLAYER_H] * self.Y_SCALE

        # --- grounded: the engine clips a standing player onto the platform, so
        # feet == platform y exactly. A crashing player sinks 2px per step and can
        # cross a platform line by chance, hence the extra stability check.
        plat_y = now[self.PLATFORM_Y] * self.Y_SCALE
        plat_active = now[self.PLATFORM_ACTIVE] > 0.5
        on_platform = jnp.any(plat_active & (jnp.abs(plat_y - feet) < 0.5))
        grounded = on_platform & (jnp.abs(feet - feet_prev) < 0.5)

        # While airborne or on a ladder no floor prop fires and the RM simply
        # keeps its current state -- the floor only updates on a confirmed landing.
        floor = 3 - jnp.searchsorted(self.FLOOR_EDGES, feet)
        on_floor_3 = grounded & (floor == 3)
        on_floor_2 = grounded & (floor == 2)
        on_floor_1 = grounded & (floor == 1)
        on_floor_0 = grounded & (floor == 0)

        # --- died: the player drops off-screen, waits 40 ticks, then respawns ---
        died = (player_y_prev >= self.CRASH_Y) & (player_y < self.CRASH_Y)

        # --- level_changed: the platform count is unique per level (4 / 18 / 20) ---
        n_platforms = jnp.sum(plat_active)
        n_platforms_prev = jnp.sum(prev[self.PLATFORM_ACTIVE] > 0.5)
        level_changed = jnp.abs(n_platforms - n_platforms_prev) > 0.5

        # --- fruit: active 1 -> 0. A level reset re-activates, i.e. 0 -> 1. ---
        fruit_prev = prev[self.FRUIT_ACTIVE] > 0.5
        fruit_collected = jnp.any(fruit_prev & (now[self.FRUIT_ACTIVE] <= 0.5))

        # --- monkey punched: state != 0 -> 0. Monkeys in state 5 climb off the
        # top on their own; a level reset zeroes all of them at once.
        monkey_now = now[self.MONKEY_STATE] * self.ID_SCALE
        monkey_prev = prev[self.MONKEY_STATE] * self.ID_SCALE
        monkey_punched = jnp.any(
            (monkey_prev > 0.5)
            & (jnp.abs(monkey_prev - 5.0) >= 0.5)
            & (monkey_now < 0.5)
        ) & jnp.logical_not(died | level_changed)

        return jnp.array([
            died,
            level_changed,
            on_floor_3,
            on_floor_2,
            on_floor_1,
            on_floor_0,
            fruit_collected,
            monkey_punched,
        ]).astype(jnp.int32)
