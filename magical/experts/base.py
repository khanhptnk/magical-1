import os
import sys

from stable_baselines3.common.policies import BasePolicy


class BaseExpert(BasePolicy):

    def __init__(self, observation_space,
                       action_space,
                       not_used_features_extractor_class):

        super().__init__(observation_space,
                         action_space,
                         not_used_features_extractor_class)

