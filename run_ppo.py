import gymnasium as gym
from common.env_util import make_vec_dmcontrol_env
#from ppo_reward.ppo import PPO
from ppo_cuda.ppo import PPO
import dmc2gym
import matplotlib.pyplot as plt
import imageio
import os
from common.vec_env import VecNormalize


from typing import Callable,  Union

def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func

seeds = [12345, 23451, 34512]
for seed in seeds:
    # ENV PARAMS
    env_name = 'walker'
    task_name = 'walk'
    num_envs = 32

    # POLICY PARAMS
    lr = 0.00005
    policy_batch = 64
    ent_coef = 0.0
    num_layers = 3
    num_hidden_dim = 256
    gae_lambda = 0.92
    clip_range = linear_schedule(0.4)
    n_steps = 500
    n_epochs = 20
    # Parallel environments
    env = make_vec_dmcontrol_env(env_name, task_name, n_envs=num_envs, seed=seed)

    print('\n\nStarting Test for ' + env_name + '_' + task_name + '\n\n')

    input_dim_a = env.action_space.shape[0]
    input_dim_obs = env.observation_space.shape[0]

    # network arch
    net_arch = dict(
        pi=[num_hidden_dim] * num_layers,
        vf=[num_hidden_dim] * num_layers
    )
    policy_kwargs = dict(net_arch=net_arch)

    env= VecNormalize(env, norm_reward=False)

    model = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log="./tests/" + env_name + "_" + task_name + "_ppo_seed_" +str(seed) + "/")
    model.learn(total_timesteps=160000)

    del model