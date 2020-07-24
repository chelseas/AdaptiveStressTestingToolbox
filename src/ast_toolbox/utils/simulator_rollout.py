"""Utility functions related to sampling."""

import time

import numpy as np

from garage.misc import tensor_utils
from gym.spaces.utils import flatten


def rollout(env,
            default_action,
            *,
            max_path_length=np.inf,
            animated=False,
            speedup=1,
            deterministic=False):
    """Sample a single rollout of the agent in the environment.

    Args:
        env: environment simulator to rollout
        default_action: to execute for environment
        max_path_length(int): If the rollout reaches this many timesteps, it is
            terminated.
        animated(bool): If true, render the environment after each step.
        speedup(float): Factor by which to decrease the wait time between
            rendered steps. Only relevant, if animated == true.
        deterministic (bool): If true, use the mean action returned by the
            stochastic policy instead of sampling from the returned action
            distribution.

    Returns:
        dict[str, np.ndarray or dict]: Dictionary, with keys:
            * observations(np.array): Flattened array of observations.
                There should be one more of these than actions. Note that
                observations[i] (for i < len(observations) - 1) was used by the
                agent to choose actions[i]. Should have shape (T + 1, S^*) (the
                unflattened state space of the current environment).
            * actions(np.array): Non-flattened array of actions. Should have
                shape (T, S^*) (the unflattened action space of the current
                environment).
            * rewards(np.array): Array of rewards of shape (T,) (1D array of
                length timesteps).
            * agent_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `agent_info` arrays.
            * env_infos(Dict[str, np.array]): Dictionary of stacked,
                non-flattened `env_info` arrays.
            * dones(np.array): Array of termination signals.

    """
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    dones = []
    o = env.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < (max_path_length or np.inf):
        o = flatten(env.observation_space, o)
        next_o, r, d, env_info = env.step(default_action)
        observations.append(o)
        rewards.append(r)
        actions.append(default_action)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        dones.append(d)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        dones=np.array(dones),
    )
