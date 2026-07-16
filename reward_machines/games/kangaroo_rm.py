"""
Simple Kangaroo Reward Machine.

This file contains only the Kangaroo-specific RM logic:
- which RM states exist
- how we detect events from old/new game state
- how the RM state changes
"""
import jax.numpy as jnp

RM_ZONE_0_BOTTOM = 0
RM_ZONE_1_LOWER_MIDDLE = 1
RM_ZONE_2_UPPER_MIDDLE = 2
RM_ZONE_3_TOP = 3
RM_POSITIVE_SCORE = 4
RM_DANGER_LOST_LIFE = 5
RM_FRUIT_COLLECTED = 6
RM_LEVEL_COMPLETED = 7

# Backward-compatible aliases
RM_NORMAL = RM_ZONE_0_BOTTOM
RM_PROGRESS = RM_ZONE_1_LOWER_MIDDLE

NUM_RM_STATES = 8


def rm_state_name(rm_state: int) -> str:
    names = {
        RM_ZONE_0_BOTTOM: "zone_0_bottom",
        RM_ZONE_1_LOWER_MIDDLE: "zone_1_lower_middle",
        RM_ZONE_2_UPPER_MIDDLE: "zone_2_upper_middle",
        RM_ZONE_3_TOP: "zone_3_top",
        RM_POSITIVE_SCORE: "positive_score",
        RM_DANGER_LOST_LIFE: "danger_lost_life",
        RM_FRUIT_COLLECTED: "fruit_collected",
        RM_LEVEL_COMPLETED: "level_completed",
    }
    return names.get(int(rm_state), "unknown")


def get_vertical_zone(y: int) -> int:
    """
    Higher zone number means higher progress in the level.

    Atari y-coordinate gets smaller when the player moves upward.

    zone 0 = bottom
    zone 1 = lower-middle
    zone 2 = upper-middle
    zone 3 = top
    """
    y = int(y)

    if y < 60:
        return 3
    if y < 100:
        return 2
    if y < 140:
        return 1
    return 0


def get_kangaroo_events_from_values(
    old_y: int,
    old_score: int,
    old_lives: int,
    new_state,
) -> dict:
    """
    Extract simple Kangaroo events.

    Old values are passed as normal Python integers because JAX may delete
    old_state arrays after env.step().
    """

    new_y = int(new_state.player.y)
    new_score = int(new_state.score)
    new_lives = int(new_state.lives)

    old_zone = get_vertical_zone(old_y)
    new_zone = get_vertical_zone(new_y)

    progress_zone_up = new_zone > old_zone
    score_increased = new_score > old_score
    lost_life = new_lives < old_lives

    return {
        "progress_zone_up": progress_zone_up,
        "score_increased": score_increased,
        "lost_life": lost_life,
        "old_zone": old_zone,
        "new_zone": new_zone,
    }

def update_kangaroo_rm_state(rm_state: int, events: dict) -> int:
    """
    Improved Kangaroo Reward Machine.

    Main idea:
    - Normally, RM state represents the current vertical zone.
    - If score increases, temporarily go to positive_score.
    - If life is lost, temporarily go to danger_lost_life.
    """

    if events["lost_life"]:
        return RM_DANGER_LOST_LIFE

    if events["score_increased"]:
        return RM_POSITIVE_SCORE

    return events["new_zone"]


#jax la uyumlu olmsı icin yani jax için yaptık
#ve PPO içinde çalışabilmesi için


def get_vertical_zone_jax(y):
    """
    JAX-compatible vertical zone function.

    Higher zone number means higher progress.
    Atari y-coordinate gets smaller when the player moves upward.
    """
    return jnp.where(
        y < 60,
        3,
        jnp.where(
            y < 100,
            2,
            jnp.where(y < 140, 1, 0),
        ),
    )


def get_kangaroo_events_jax(old_state, new_state):
    """
    JAX-compatible Kangaroo event extraction.

    This version is used inside PPO training with jax.jit / lax.scan.
    """

    old_zone = get_vertical_zone_jax(old_state.player.y)
    new_zone = get_vertical_zone_jax(new_state.player.y)

    progress_zone_up = new_zone > old_zone
    score_increased = new_state.score > old_state.score
    lost_life = new_state.lives < old_state.lives

    old_fruit_count = jnp.sum(old_state.level.fruit_actives)
    new_fruit_count = jnp.sum(new_state.level.fruit_actives)
    fruit_collected = new_fruit_count < old_fruit_count

    level_completed = jnp.logical_and(
        jnp.logical_not(old_state.level_finished),
        new_state.level_finished,
    )

    return (
        progress_zone_up,
        score_increased,
        lost_life,
        fruit_collected,
        level_completed,
        new_zone,
    )


def update_kangaroo_rm_state_jax(
        rm_state,
        progress_zone_up,
        score_increased,
        lost_life,
        fruit_collected,
        level_completed,
        new_zone,
):
    next_rm_state = new_zone

    next_rm_state = jnp.where(
        score_increased,
        RM_POSITIVE_SCORE,
        next_rm_state,
    )

    next_rm_state = jnp.where(
        fruit_collected,
        RM_FRUIT_COLLECTED,
        next_rm_state,
    )

    next_rm_state = jnp.where(
        level_completed,
        RM_LEVEL_COMPLETED,
        next_rm_state,
    )

    next_rm_state = jnp.where(
        lost_life,
        RM_DANGER_LOST_LIFE,
        next_rm_state,
    )

    return next_rm_state