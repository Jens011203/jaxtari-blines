# PPO + Reward Machine Progress Notes

## Project Context

The practical project focuses on reimplementing neuro-symbolic baselines for JAXAtari using Reward Machines.
The current work is an initial integration of Reward Machine states into the existing object-based PPO baseline for Pong.

## Current Goal

The goal of this step was not to solve Pong or achieve a high score.
The goal was to verify that a PPO baseline can be extended with Reward Machine state information and still run successfully on JAXAtari.

## Files Used / Modified

### Existing PPO baseline

```text
scripts/benchmarks/ppo_jaxatari_scan.py
```

This file contains the already implemented PPO baseline.

### New PPO + RM file

```text
scripts/benchmarks/ppo_jaxatari_scan_rm.py
```

This file was created as a copy of the PPO baseline and then modified to include Reward Machine state information.

### Test config

```text
scripts/benchmarks/config/alg/ppo_jaxatari_object_test.yaml
```

This config was used for a short object-based PPO test on Pong.

Important settings:

```yaml
ENV_ID: pong
PIXEL_BASED: False
TOTAL_TIMESTEPS: 100000
NUM_ENVS: 8
NUM_STEPS: 128
TRACK: False
CAPTURE_VIDEO: False
SAVE_MODEL: False
```

## Implemented Reward Machine Idea

For Pong, a very simple Reward Machine was used with three states:

```text
0 = normal
1 = scored
2 = conceded
```

The Reward Machine state is updated based on the environment reward:

```text
reward > 0  → scored
reward < 0  → conceded
reward == 0 → no special event
```

Since Pong has very sparse and simple reward events, this Reward Machine is only a first technical integration test.

## One-Hot Encoding of RM State

The RM state is represented as a one-hot vector:

```text
0 → [1, 0, 0]
1 → [0, 1, 0]
2 → [0, 0, 1]
```

This one-hot RM state is appended to the flattened object-centric observation.

Conceptually:

```text
new_observation = flattened_pong_observation + one_hot(rm_state)
```

## Main Code Changes

### 1. Added RM helper functions

```python
NUM_RM_STATES = 3

def update_rm_state(rm_state, reward):
    scored = reward > 0
    conceded = reward < 0

    next_rm_state = jnp.where(scored, 1, rm_state)
    next_rm_state = jnp.where(conceded, 2, next_rm_state)

    reset_to_normal = jnp.logical_or(rm_state == 1, rm_state == 2)
    next_rm_state = jnp.where(reset_to_normal, 0, next_rm_state)

    return next_rm_state


def append_rm_state(obs, rm_state):
    rm_one_hot = jax.nn.one_hot(rm_state, NUM_RM_STATES)
    return jnp.concatenate([obs, rm_one_hot], axis=-1)
```

### 2. Adjusted network initialization

The PPO network was initialized with an observation that already includes the appended RM state.
This is necessary because the input dimension increases by 3.

### 3. Added RM state to the rollout carry

The rollout carry was extended from:

```text
(agent_state, obs, done, key, env_state)
```

to:

```text
(agent_state, obs, done, key, env_state, rm_state)
```

This allows the Reward Machine state to be carried from one environment step to the next.

### 4. Updated RM state during environment interaction

Inside `step_once`, the raw environment observation is first received from the environment.
Then the RM state is updated based on the reward, and the updated RM state is appended to the next observation.

Conceptually:

```text
raw_next_obs, reward = env.step(...)
next_rm_state = update_rm_state(rm_state, reward)
next_obs = append_rm_state(raw_next_obs, next_rm_state)
```

### 5. Added RM state logging

To verify that RM states are actually changing, the distribution of RM states across the 8 parallel environments is logged:

```text
rm_counts=[normal_count, scored_count, conceded_count]
```

Example:

```text
rm_counts=[7, 0, 1]
```

This means:

```text
7 environments are in normal state
0 environments are in scored state
1 environment is in conceded state
```

## Test Result

The modified PPO + RM file was successfully run for 100,000 timesteps.

Example output:

```text
Compile + first iteration time: 2.21 seconds.
Iteration    1 | global_step=   1024 | avg_return=0.00 | avg_length=0.00 | rm_counts=[7, 0, 1] | loss=-0.0034 | SPS=431
Iteration   10 | global_step=  10240 | avg_return=-20.00 | avg_length=895.62 | rm_counts=[8, 0, 0] | loss=0.0003 | SPS=2208
Iteration   90 | global_step=  92160 | avg_return=-19.49 | avg_length=1007.04 | rm_counts=[8, 0, 0] | loss=-0.0008 | SPS=7338
Training done.
Total train time: 13.23 seconds / 0.22 minutes.
Process finished with exit code 0
```

## Interpretation

The result shows that:

```text
PPO + RM integration runs successfully.
The RM state is appended to the observation.
The RM state is carried through the rollout.
The RM state can change during training.
Training finishes without shape or carry errors.
```

The average return remains around -19 to -20.
This is expected because 100,000 timesteps are only a short technical test and Pong is not ideal for a rich Reward Machine.

## Limitations

Pong is not an ideal game for Reward Machines because it has very few meaningful subgoals.
The current RM mainly models two events:

```text
scored
conceded
```

There are no rich intermediate objectives like collecting oxygen, rescuing divers, or reaching a specific object.

Therefore, this Pong RM should be seen as a technical proof of concept rather than a strong performance-improving Reward Machine.

## Next Steps

Possible next steps:

```text
1. Compare normal PPO and PPO + RM for 1,000,000 timesteps.
2. Integrate the RM logic with the group’s RMWrapper structure.
3. Move to a game with richer subgoals, such as Seaquest.
4. Define more meaningful RM states and events for Seaquest.
5. Investigate whether Counterfactual Reward Machines require an off-policy algorithm.
```

## Current Status

Minimum technical integration is successful:

```text
Normal PPO baseline runs.
PPO + RM version runs.
RM state is appended to the flattened object-centric observation.
RM state is carried through rollout.
Training starts and finishes successfully on Pong.
```








## 1M Timestep Comparison Results
we are comparing normal PPO Baseline and PPO+Reward Machine

### Normal PPO Baseline
Configuration:
```text
ENV_ID: pong
PIXEL_BASED: False
TOTAL_TIMESTEPS: 1,000,000
NUM_ENVS: 8
NUM_STEPS: 128
TRACK: False
CAPTURE_VIDEO: False
SAVE_MODEL: False

RESULT
Training finished successfully.
Process finished with exit code 0.
Total train time: 183.41 seconds / 3.06 minutes
Final logged avg_return: -7.26
Best visible avg_return: approximately -6.72
Final visible global_step: 993,280

INTERPRETATION
The normal PPO baseline successfully learned compared to the initial performance.
At the beginning, the avg_return was around -19 to -20.
Towards the end, the avg_return improved to around -7.
This shows that the object-based PPO baseline is working and learning on Pong.
```

### PPO + Reward Machine

Configuration:

```text

ENV_ID: pong
PIXEL_BASED: False
TOTAL_TIMESTEPS: 1,000,000
NUM_ENVS: 8
NUM_STEPS: 128
TRACK: False
CAPTURE_VIDEO: False
SAVE_MODEL: False

RESULT

Training finished successfully.
Process finished with exit code 0.
Total train time: 117.57 seconds / 1.96 minutes
Final logged avg_return: -7.25
Best visible avg_return: approximately -7.25
Final visible global_step: 993,280
RM state logging: rm_counts showed state changes such as [7, 0, 1], [6, 0, 2], and [7, 1, 0].
```

### Comparasion Summary
Normal PPO:
- Final logged avg_return: -7.26
- Best visible avg_return: approximately -6.72
- Total train time: 183.41 seconds / 3.06 minutes

PPO + RM:
- Final logged avg_return: -7.25
- Best visible avg_return: approximately -7.25
- Total train time: 117.57 seconds / 1.96 minutes
- RM state was successfully carried and updated during rollout.

Both normal PPO and PPO + RM learned on Pong and improved from around -20 avg_return to around -7 avg_return.
The PPO + RM implementation runs successfully and the RM state changes during training.
However, PPO + RM does not clearly outperform normal PPO on Pong in this first comparison.
This is expected because Pong has only simple scored/conceded events and very few meaningful subgoals for a Reward Machine.
The result should therefore be interpreted as a successful technical proof of concept rather than a strong performance improvement.





## Kangaroo PPO + Reward Machine Progress

### Goal

The goal was to implement and test a first Reward Machine setup for Kangaroo and integrate it into the PPO training loop.

### Implemented steps


- Inspected the Kangaroo environment and its object-centric state.
- Identified useful Kangaroo state information:
  - `player.y`
  - `score`
  - `lives`
  - `level_finished`
  - `fruit_actives`
- Implemented a Kangaroo-specific Reward Machine in:

  `reward_machines/games/kangaroo_rm.py`

- Added JAX-compatible Reward Machine functions for PPO training:
  - `get_kangaroo_events_jax`
  - `update_kangaroo_rm_state_jax`
- Integrated the Kangaroo Reward Machine into a PPO+RM training file:

  `scripts/benchmarks/ppo_jaxatari_scan_rm_kangaroo.py`

- Added the RM state as a one-hot vector to the object-based observation.
- Debugged the JAXAtari wrapper state structure and found that the real Kangaroo state can be accessed through:

  `LogState -> ObjectCentricState -> AtariState -> KangarooState`
---


### First Reward Machine version

The first Kangaroo RM used four states:

0 = normal
1 = progress
2 = positive_score
3 = danger_lost_life


The detected events were:
progress_zone_up
score_increased
lost_life

Result for 100k timesteps:
PPO+RM ran successfully.
The RM state was updated during training.
Example rm_counts: [7, 1, 0, 0], [6, 2, 0, 0]

Best visible avg_return: 57.03
Final visible avg_return: 0.00
Total train time: 79.83 seconds
Improved zone-based Reward Machine

The improved Kangaroo RM uses six states:

0 = zone_0_bottom
1 = zone_1_lower_middle
2 = zone_2_upper_middle
3 = zone_3_top
4 = positive_score
5 = danger_lost_life

Main idea:

Normally, the RM state represents the current vertical zone of the player.
If the score increases, the RM temporarily enters positive_score.
If a life is lost, the RM temporarily enters danger_lost_life.

Result for 100k timesteps:

PPO+RM ran successfully.
The RM state was updated correctly during training.
Example rm_counts: [6, 2, 0, 0, 0, 0], [3, 5, 0, 0, 0, 0]

Best visible avg_return: 75.00
Final visible avg_return: 25.00
Total train time: 49.91 seconds


Baseline comparison: normal PPO
Normal PPO was also tested on Kangaroo for 100k timesteps.
Result:
Best visible avg_return: 265.23
Final visible avg_return: 265.23
Total train time: 80.26 seconds

### Initial 100k comparison

Method	     Best visible avg_return	Final visible avg_return	Train time
Normal PPO  	265.23	                265.23	                    80.26 sec
PPO + first RM	57.03	                0.00	                    79.83 sec
PPO + improved  75.00	                25.00	                    49.91 sec
zone-based RM
	
### Interpretation

The Kangaroo Reward Machine integration works technically. The RM state is updated during PPO training and is appended to the object-based observation as a one-hot vector.
The improved zone-based RM gives more meaningful state information than the first event-only RM. Instead of only representing short events, it also represents the player's current vertical progress in the level.
However, in the initial 100k timestep test, normal PPO still performed better than both PPO+RM versions.
These are only short technical tests with one seed, so the results should not be interpreted as final performance conclusions.

### Next steps

### Future work

Test multiple random seeds to make the comparison more robust.

Improve the Kangaroo Reward Machine with more game-specific events, for example:

- fruit collection
- reaching higher platforms
- level completion
- avoiding life loss
- reaching or approaching the child

Consider adding reward shaping based on RM events.

Later, compare DQN/DQN+RM if the DQN setup is available.

### Kangaroo 1M comparison

| Method            | Best visible avg_return | Final visible avg_return | Train time |
| Normal PPO                   | 2307.62      | 1725.00                  | 467.88 sec |
| PPO + improved zone-based RM | 1550.00      | 1066.99                  | 427.44 sec |

In the 1M timestep experiment, normal PPO still performed better than PPO+RM. The PPO+RM integration works technically and the RM state is updated during training, but the current zone-based RM does not yet improve performance over the normal PPO baseline.

The RM counts show that most environments remain in the lower zones, mainly `zone_0_bottom` and `zone_1_lower_middle`, while higher zones and score/life related RM states are rarely reached.
The current Kangaroo PPO+RM integration works technically, but it does not outperform normal PPO because the RM is only appended as additional state information and does not yet provide stronger reward shaping or counterfactual learning signals. Moreover, most RM states remain in the lower vertical zones, so the current RM design provides limited useful guidance.

### Kangaroo PPO+RM v2

After the first PPO+RM version, we extended the Kangaroo Reward Machine with two additional game-specific events:

- `fruit_collected`
- `level_completed`

These events are based on the existing Kangaroo environment state variables `fruit_actives` and `level_finished`.

We then ran a 1M timestep experiment with the improved RM v2.

| Method | Best avg_return | Final avg_return |
|---|---:|---:|
| PPO+RM v1 | 1550.00 | 1066.99 |
| PPO+RM v2 | 1800.00 | 1450.00 |
| Normal PPO | 2307.62 | 1725.00 |

The improved PPO+RM v2 performs better than the first PPO+RM version, but it still does not outperform the normal PPO baseline. The RM count logs confirm that the 8-state RM is integrated correctly, although most environments still remain in the lower zone states during training.





















































