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
        return self.venv.get_attr(attr_name, indices=[id])[0]

    def env_method(self, id, method_name, *method_args, **method_kwargs):
        return self.venv.env_method(method_name, *method_args, indices=[id], **method_kwargs)[0]
