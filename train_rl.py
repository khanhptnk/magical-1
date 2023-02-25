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
import magical.entities as en

BATCH_SIZE = 32
IMAGE_FEAT_DIM = 256
TARGET_FEAT_DIM = 16


class MAGICALNet(nn.Module):
    """Custom CNN for MAGICAL policies."""
    def __init__(self, observation_space, train_envs, test_envs,
            image_dim=256, target_dim=16, width=2):

        super().__init__()

        self.train_envs = train_envs
        self.test_envs = test_envs

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
            nn.Linear(fc_in_size, image_dim),
            # Stable Baselines will add extra affine layer on top of this reLU
            nn.ReLU(),
        ]
        layers = [*conv_layers, *reduction_layers]
        self.image_feature_layer = nn.Sequential(*layers)

        self.position_feature_layer = nn.Linear(2, TARGET_FEAT_DIM)
        self.colour_embeddings = nn.Embedding(len(en.SHAPE_COLOURS), TARGET_FEAT_DIM)
        self.shape_embeddings  = nn.Embedding(len(en.SHAPE_TYPES), TARGET_FEAT_DIM)

        self.features_dim = image_dim + target_dim * 3

    def forward(self, x, traj_info=None):
        img_feat = self.image_feature_layer(x)
        device = x.device
        target_position, target_colour, target_shape_type = [], [], []
        envs = self.train_envs if hasattr(self.train_envs[0], 'target_position') \
            else self.test_envs
        for e in envs:
            target_position.append(e.target_position.tolist())
            target_colour.append(en.SHAPE_COLOURS.tolist().index(e.target_colour_name))
            target_shape_type.append(en.SHAPE_TYPES.tolist().index(e.target_shape_type))
        target_position = torch.tensor(target_position).to(device).float()
        target_colour  = torch.tensor(target_colour).to(device).long()
        target_shape_type = torch.tensor(target_shape_type).to(device).long()

        position_feat = self.position_feature_layer(target_position)
        colour_feat = self.colour_embeddings(target_colour)
        shape_feat = self.shape_embeddings(target_shape_type)

        print(img_feat.shape, position_feat.shape, colour_feat.shape, shape_feat.shape)

        output = torch.cat([img_feat, position_feat, colour_feat, shape_feat], dim=-1)

        return output


#test_env = Monitor(gym.make('PickAndPlace-Test-LoRes4A-v0'))

test_env = DummyVecEnv(
    [lambda: Monitor(gym.make('PickAndPlace-Test-LoRes4A-v0', debug_reward=True)) for _ in range(1)])


train_env = DummyVecEnv(
    [lambda: Monitor(gym.make('PickAndPlace-Demo-LoRes4A-v0', debug_reward=True)) for _ in range(1)])

model_name = 'pick_and_place_ppo'
exp_dir = 'experiments/pick_and_place_ppo_1m'

feature_dim = 256 + TARGET_FEAT_DIM * 3

policy_kwargs = dict(
    features_extractor_class=MAGICALNet,
    features_extractor_kwargs=dict(train_envs=train_env.envs,
                                   test_envs=test_env.envs,
                                   image_dim=IMAGE_FEAT_DIM,
                                   target_dim=TARGET_FEAT_DIM),
    net_arch=dict(pi=[feature_dim, 32], vf=[feature_dim, 32])
)

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

