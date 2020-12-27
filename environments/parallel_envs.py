"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import os

import gym
import numpy as np
import torch

from environments.wrappers import TimeLimitMask
from environments.wrappers import VariBadWrapper
from utils import bench
from utils.common.vec_env import VecEnvWrapper
from utils.common.vec_env.dummy_vec_env import DummyVecEnv
from utils.common.vec_env.subproc_vec_env import SubprocVecEnv
from utils.common.vec_env.vec_normalize import VecNormalize as VecNormalize_
from metaworld.benchmarks import ML1

class MetaWorldMask(gym.Wrapper):

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if not info['success']:
            rew = 0
        info['task']=0

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def get_task(self):
        return self.env.tasks_pool[np.random.randint(0,50)]

    def sample_task(self):
        return self.env.tasks_pool[np.random.randint(0,50)]

    def reset_task(self, task):
        if task is None:
            task = self.sample_task()
        self.env.set_task(task)

def make_env(env_id, seed, rank, log_dir, allow_early_resets,
             episodes_per_task, **kwargs):
    def _thunk():
        if not 'metaworld' in env_id:
            env = gym.make(env_id, **kwargs)
            if seed is not None:
                env.seed(seed + rank)

            if str(env.__class__.__name__).find('TimeLimit') >= 0:
                env = TimeLimitMask(env)
        else:
            available_tasks = ML1.available_tasks()
            print(available_tasks)
            env = ML1.get_train_tasks(available_tasks[16])
            print(env)
            env._max_episode_steps=150
            tasks = env.sample_tasks(50)
            env.tasks_pool = tasks

            def get_task():
                return self.tasks_pool[np.random.randint(0, 50)]
            env.get_task=get_task
            env=MetaWorldMask(env)
            #def reset_task():
            #    pass
            #env.reset_task = reset_task


        env = VariBadWrapper(env=env, episodes_per_task=episodes_per_task)

        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)),
                                allow_early_resets=allow_early_resets)

        return env

    return _thunk


def make_vec_envs(env_name, seed, num_processes, gamma, log_dir,
                  device, allow_early_resets, episodes_per_task,
                  obs_rms, ret_rms, rank_offset=0,
                  **kwargs):
    """
    :param obs_rms: running mean and std for observations
    :param ret_rms: running return and std for rewards
    """
    envs = [make_env(env_id=env_name, seed=seed, rank=rank_offset + i, log_dir=log_dir,
                     allow_early_resets=allow_early_resets,
                     episodes_per_task=episodes_per_task, **kwargs)
            for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, obs_rms=obs_rms, ret_rms=ret_rms, ret=False)
        else:
            envs = VecNormalize(envs, obs_rms=obs_rms, ret_rms=ret_rms, gamma=gamma)

    envs = VecPyTorch(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset_mdp(self, index=None):
        obs = self.venv.reset_mdp(index=index)
        if isinstance(obs, list):
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def reset(self, index=None, task=None):
        if task is not None:
            assert isinstance(task, list)
        obs = self.venv.reset(index=index, task=task)
        if isinstance(obs, list):
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        actions = actions.squeeze(1).cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, list):  # raw + normalised
            obs = [torch.from_numpy(o).float().to(self.device) for o in obs]
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        if isinstance(reward, list):  # raw + normalised
            reward = [torch.from_numpy(r).unsqueeze(dim=1).float().to(self.device) for r in reward]
        else:
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float().to(self.device)
        return obs, reward, done, info

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
        """

        if attr in ['num_states', '_max_episode_steps']:
            return self.unwrapped.get_env_attr(attr)

        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr


class VecNormalize(VecNormalize_):

    def __init__(self, envs, obs_rms, ret_rms, *args, **kwargs):
        super(VecNormalize, self).__init__(envs, obs_rms=obs_rms, ret_rms=ret_rms, *args, **kwargs)
        self.training = True

    def _obfilt(self, obs):
        if self.training:
            self.obs_rms.update(obs)
        obs_norm = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clipobs,
                           self.clipobs)
        return [obs, obs_norm]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __getattr__(self, attr):
        """
        If env does not have the attribute then call the attribute in the wrapped_env
        """
        try:
            orig_attr = self.__getattribute__(attr)
        except AttributeError:
            orig_attr = self.unwrapped.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr
