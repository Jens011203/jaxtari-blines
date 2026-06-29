from typing import Callable
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxatari.environment import JaxEnvironment
from jaxatari.wrappers import JaxatariWrapper


def evaluate_dqn(
    network_params,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    QNetwork: nn.Module,
    seed: int = 1,
    use_rm: bool = False,
):
    env: JaxEnvironment | JaxatariWrapper = make_env(env_id)()
    key = jax.random.key(seed)

    @jax.jit
    def wrapped_reset(key):
        next_obs, state = env.reset(key)
        return next_obs.squeeze()[None, ...], state

    @jax.jit
    def wrapped_step(state, action):
        """wrappes the step function of the environment to correct the observation shape"""
        next_obs, next_state, reward, terminated, truncated, info = env.step(state, action.squeeze())
        done = jnp.logical_or(terminated, truncated)
        return next_obs.squeeze()[None, ...], next_state, reward, done, info

    network = QNetwork(action_dim=env.action_space().n)

    @jax.jit
    def get_action(params, obs):
        q_vals = network.apply(params, obs)
        return jnp.argmax(q_vals, axis=-1)

    def step_fn(carry, _):
        next_obs, env_state = carry
        actions = jax.vmap(get_action, in_axes=(None, 0))(network_params, next_obs)
        next_obs, env_state, reward, done, infos = jax.vmap(wrapped_step)(env_state, jnp.array(actions))
        first_states = jax.tree.map(lambda x: x[0], env_state)
        reward = infos["env_reward"]
        return (next_obs, env_state), (first_states, done, reward, actions)

    reset_keys = jax.random.split(key, eval_episodes)
    next_obs, env_states = jax.vmap(wrapped_reset)(reset_keys)

    carry = (next_obs, env_states, reset_keys)
    all_first_states = []
    all_dones = []
    all_rewards = []
    done_ever = jnp.zeros(eval_episodes, dtype=jnp.bool_)

    @jax.jit
    def scanned_step(carry):
        carry, (first_states_chunk, dones_chunk, rewards_chunk, actions_chunk) = jax.lax.scan(
            step_fn, carry, None, length=1000
        )
        return carry, (first_states_chunk, dones_chunk, rewards_chunk, actions_chunk)

    while not jnp.all(done_ever):
        carry, (first_states_chunk, dones_chunk, rewards_chunk, actions_chunk) = scanned_step(carry)
        all_first_states.append(first_states_chunk)
        all_dones.append(dones_chunk)
        all_rewards.append(rewards_chunk)
        done_ever = done_ever | jnp.any(dones_chunk, axis=0)

    first_states_history = jax.tree.map(
        lambda *xs: jnp.concatenate(xs, axis=0), *all_first_states
    )
    dones = jnp.concatenate(all_dones, axis=0)
    rewards = jnp.concatenate(all_rewards, axis=0)

    # mask everything after the first done per episode, then sum the reward
    first_done = jnp.argmax(dones, axis=0)
    has_finished = jax.lax.cummax(dones.astype(jnp.int32), axis=0)
    mask_after_first_done = jnp.pad(has_finished[:-1, :], ((1, 0), (0, 0)), constant_values=0)
    masked_rewards = rewards * (1 - mask_after_first_done)
    episodic_returns = jnp.sum(masked_rewards, axis=0) 
    print(f"Evaluated {eval_episodes} episodes, mean return: {episodic_returns.mean():.2f}, std return: {episodic_returns.std():.2f}")

    # env states of the first episode (for video), same nesting path as PPO eval
    if use_rm:
        state = first_states.atari_state.env_state.atari_state.env_state
    else:
        state = first_states.atari_state.atari_state.env_state
    env_states_until_done = jax.tree.map(lambda x: x[:first_done[0] + 1], state)

    return episodic_returns, env_states_until_done
