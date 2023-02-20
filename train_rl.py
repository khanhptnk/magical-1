import os
import sys
import torch
import torch.nn as nn
import numpy as np

import gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import magical
magical.register_envs()


class MAGICALNet(nn.Module):
    """Custom CNN for MAGICAL policies."""
    def __init__(self, observation_space, out_chans=256, width=2):

        super().__init__()

        w = width
        def conv_block(i, o, k, s, p, b=False):
            return [
                # batch norm has its own bias, so don't add one to conv layers by default
                nn.Conv2d(i, o, kernel_size=k, stride=s, padding=p, bias=b,
                          padding_mode='zeros'),
                nn.ReLU(),
                nn.BatchNorm2d(o)
            ]
        conv_layers = [
            *conv_block(i=observation_space.shape[0], o=32*w, k=5, s=1, p=2, b=True),
            *conv_block(i=32*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
        ]
        # final FC layer to make feature maps the right size
        test_tensor = torch.zeros((1,) + observation_space.shape)
        for layer in conv_layers:
            test_tensor = layer(test_tensor)
        fc_in_size = np.prod(test_tensor.shape)
        reduction_layers = [
            nn.Flatten(),
            nn.Linear(fc_in_size, out_chans),
            # Stable Baselines will add extra affine layer on top of this reLU
            nn.ReLU(),
        ]
        self.features_dim = out_chans
        all_layers = [*conv_layers, *reduction_layers]
        self.feature_generator = nn.Sequential(*all_layers)

    def forward(self, x, traj_info=None):
        return self.feature_generator(x)

policy_kwargs = dict(
    features_extractor_class=MAGICALNet,
    net_arch=dict(pi=[256, 256], vf=[256, 256])
)

env = gym.make('MoveToCorner-Demo-LoRes4E-v0', debug_reward=True)

model = PPO("CnnPolicy", env, n_steps=100, policy_kwargs=policy_kwargs, verbose=1)
model.learn(100000)
