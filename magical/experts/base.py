import os
import sys

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


class BaseExpert(BasePolicy):

    def __init__(self, observation_space,
                       action_space,
                       not_used_features_extractor_class):

        super().__init__(observation_space,
                         action_space,
                         not_used_features_extractor_class)

    def get_env_attr(self, id, attr_name):
        if isinstance(self.venv, SubprocVecEnv):
            remote = self.venv.remotes[id]
            remote.send(('get_attr', attr_name))
            return remote.recv()
        env = self.venv.envs[id]
        return getattr(env, attr_name)

    def env_method(self, id, method_name, *method_args, **method_kwargs):
        if isinstance(self.venv, SubprocVecEnv):
            remote = self.venv.remotes[id]
            remote.send(('env_method', (method_name, method_args, method_kwargs)))
            return remote.recv()
        env = self.venv.envs[id]
        return getattr(env, method_name)(*method_args, **method_kwargs)
