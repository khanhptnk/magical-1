import os
import sys
import torch
import torch.nn as nn
import numpy as np

import gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback


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
                #nn.BatchNorm2d(o)
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
    net_arch=dict(pi=[256, 32], vf=[256, 32])
)

test_env = Monitor(gym.make('PickAndPlace-Test-LoRes4A-v0'))

train_env = DummyVecEnv(
    [lambda: Monitor(gym.make('PickAndPlace-Demo-LoRes4A-v0', debug_reward=True)) for _ in range(32)])

model_name = 'pick_and_place_ppo'
exp_dir = 'experiments/pick_and_place_ppo_1m'

model = PPO(
    "MlpPolicy",
    train_env,
    batch_size=40,
    n_steps=80,
    policy_kwargs=policy_kwargs,
    verbose=2,
    tensorboard_log=exp_dir)

eval_callback = EvalCallback(test_env, best_model_save_path=exp_dir,
                             log_path=exp_dir, eval_freq=500,
                             deterministic=True, render=False)

model.learn(1000000, callback=eval_callback)

