import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RM_GAMES_DIR = os.path.join(PROJECT_ROOT, "reward_machines", "games")
sys.path.insert(0, RM_GAMES_DIR)



import jax
import jaxatari

from kangaroo_rm import (
    RM_NORMAL,
    rm_state_name,
    get_kangaroo_events_from_values ,
    update_kangaroo_rm_state,
)


def print_key_state(step, action, reward, done, state, rm_state, events=None):
    print(f"\nStep {step}")
    print("Action:", action)
    print("Reward:", int(reward))
    print("Done:", bool(done))

    print("player_x:", int(state.player.x))
    print("player_y:", int(state.player.y))

    if events is not None:
        print("old_zone:", events["old_zone"])
        print("new_zone:", events["new_zone"])

    print("is_jumping:", bool(state.player.is_jumping))
    print("is_climbing:", bool(state.player.is_climbing))
    print("is_crashing:", bool(state.player.is_crashing))

    print("score:", int(state.score))
    print("lives:", int(state.lives))
    print("current_level:", int(state.current_level))
    print("level_finished:", bool(state.level_finished))

    print("fruit_actives:", state.level.fruit_actives)
    print("child_position:", state.level.child_position)

    print("RM state:", rm_state, "-", rm_state_name(rm_state))


def main():
    env = jaxatari.make("kangaroo")
    key = jax.random.PRNGKey(0)

    obs, state = env.reset(key)

    rm_state = RM_NORMAL

    print("Kangaroo environment reset successful")
    print_key_state(0, "-", 0, False, state, rm_state)

    print("\n================ RANDOM STEPS WITH IMPORTED KANGAROO RM ================")

    action_key = key

    event_counts = {
        "progress_zone_up": 0,
        "score_increased": 0,
        "lost_life": 0,
    }

    rm_transition_counts = {
        "normal_to_progress": 0,
        "normal_to_positive_score": 0,
        "normal_to_danger_lost_life": 0,
    }

    for step in range(1, 101):
        action_key, subkey = jax.random.split(action_key)

        old_y = int(state.player.y)
        old_score = int(state.score)
        old_lives = int(state.lives)

        action = env.action_space().sample(subkey)

        obs, state, reward, done, info = env.step(state, action)

        events = get_kangaroo_events_from_values(
            old_y,
            old_score,
            old_lives,
            state,
        )
        for event_name in event_counts:
            if events[event_name]:
                event_counts[event_name] += 1

        old_rm_state = rm_state
        rm_state = update_kangaroo_rm_state(rm_state, events)

        if old_rm_state == 0 and rm_state == 1:
            rm_transition_counts["normal_to_progress"] += 1

        if old_rm_state == 0 and rm_state == 2:
            rm_transition_counts["normal_to_positive_score"] += 1

        if old_rm_state == 0 and rm_state == 3:
            rm_transition_counts["normal_to_danger_lost_life"] += 1

        print_key_state(step, int(action), reward, done, state, rm_state, events)

        print("EVENT progress_zone_up:", events["progress_zone_up"])
        print("EVENT score_increased:", events["score_increased"])
        print("EVENT lost_life:", events["lost_life"])

        print("RM transition:", rm_state_name(old_rm_state), "->", rm_state_name(rm_state))

        if bool(done):
            print("Episode finished.")
            break

    print("\n================ EVENT COUNTS ================")
    for name, count in event_counts.items():
        print(name, "=", count)

    print("\n================ RM TRANSITION COUNTS ================")
    for name, count in rm_transition_counts.items():
        print(name, "=", count)


if __name__ == "__main__":
    main()