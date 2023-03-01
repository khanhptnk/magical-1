import os
import sys
import torch
import torch.nn as nn
import numpy as np

import gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed


import magical
magical.register_envs()
import magical.entities as en
from magical.models.impala import ImpalaCNNFeatureExtractor

BATCH_SIZE = 32
OB_FEAT_DIM = 256
TARGET_FEAT_DIM = 32

"""
class MAGICALNet(nn.Module):
    def __init__(self, observation_space, ob_feat_dim=256, target_feat_dim=None):

        super().__init__()

        self.observation_encoder = ImpalaCNNFeatureExtractor(observation_space['past_obs'].shape)

        self.object_position_layer = nn.Linear(2, target_feat_dim)
        self.goal_position_layer = nn.Linear(2, target_feat_dim)
        self.colour_embeddings = nn.Embedding(len(en.SHAPE_COLOURS), target_feat_dim)
        self.shape_embeddings  = nn.Embedding(len(en.SHAPE_TYPES), target_feat_dim)

        self.features_dim = 256

        self.feature_layer = nn.Sequential(
            nn.Linear(ob_feat_dim + target_feat_dim * 4, self.features_dim),
            nn.ReLU()
        )

    def forward(self, x, traj_info=None):
        ob_feat = self.observation_encoder(x['past_obs'])
        colour_feat = self.colour_embeddings(x['target_colour'].squeeze(-1).long())
        shape_feat = self.shape_embeddings(x['target_type'].squeeze(-1).long())
        object_position_feat = self.object_position_layer(x['object_position'])
        goal_position_feat = self.goal_position_layer(x['goal_position'])

        inputs = (ob_feat, colour_feat, shape_feat, object_position_feat, goal_position_feat)
        inputs = torch.cat(inputs, dim=-1)
        features = self.feature_layer(inputs)

        return features
"""

class MAGICALNet(nn.Module):
    """Custom CNN for MAGICAL policies."""
    def __init__(self, observation_space, image_dim=256, target_dim=16, width=2):

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

        img_shape = observation_space['past_obs'].shape

        conv_layers = [
            *conv_block(i=img_shape[0], o=32*w, k=5, s=1, p=2, b=True),
            *conv_block(i=32*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
            *conv_block(i=64*w, o=64*w, k=3, s=2, p=1),
        ]
        # final FC layer to make feature maps the right size
        test_tensor = torch.zeros((1,) + img_shape)
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

        visual_feat = self.image_feature_layer(x['past_obs'])
        colour_feat = self.colour_embeddings(x['target_colour'].squeeze(-1).long())
        shape_feat = self.shape_embeddings(x['target_type'].squeeze(-1).long())
        position_feat = self.position_feature_layer(x['goal_position'])

        #print(visual_feat.shape, colour_feat.shape, shape_feat.shape, position_feat.shape)

        output = torch.cat([visual_feat, colour_feat, shape_feat, position_feat], dim=-1)

        return output


def make_env(rank, make_fn, seed=0):
    def _init():
        env = make_fn()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':

    test_env = DummyVecEnv(
        [lambda: Monitor(gym.make('PickAndPlace-Test-LoResCHW4A-v0',
            rand_shape_colour=False,
            rand_shape_type=False,
            debug_reward=True)) for _ in range(1)]
    )

    num_cpu = 8
    make_train_env_fn = lambda: Monitor(gym.make('PickAndPlace-Demo-LoResCHW4A-v0',
                                        rand_shape_colour=False,
                                        rand_shape_type=False,
                                        debug_reward=True)
    )

    train_env = SubprocVecEnv([make_env(i, make_train_env_fn) for i in range(num_cpu)])

    #exp_dir = 'experiments/pick_hard_hidden_512_ppo_1m'
    #exp_dir = 'experiments/pick_with_true_direction_ppo_1m'

    #exp_dir = 'experiments/pick_and_place_ppo_1m'

    exp_dir = 'experiments/pick_impala_ppo_1m'

    #pretrain_dir = 'experiments/pick_with_true_direction_ppo_1m'


    feature_dim = OB_FEAT_DIM + TARGET_FEAT_DIM * 4

    """
    policy_kwargs = dict(
        features_extractor_class=MAGICALNet,
        features_extractor_kwargs=dict(ob_feat_dim=OB_FEAT_DIM,
                                    target_feat_dim=TARGET_FEAT_DIM),
        net_arch=dict(pi=[feature_dim, 32], vf=[feature_dim, 32])
    )
    """

    policy_kwargs = dict(
        features_extractor_class=MAGICALNet,
        features_extractor_kwargs=dict(image_dim=OB_FEAT_DIM,
                                       target_dim=TARGET_FEAT_DIM),
        net_arch=dict(pi=[feature_dim, 32], vf=[feature_dim, 32])
    )

    model = PPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=5e-4,
        n_steps=80,
        batch_size=64,
        #n_epochs=3,
        #gamma=0.999,
        #gae_lambda=0.95,
        #clip_range=0.2,
        #clip_range_vf=0.2,
        #normalize_advantage=True,
        #ent_coef=0.01,
        #vf_coef=0.5,
        #max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=2,
        tensorboard_log=exp_dir,
        device=torch.device('cuda', 1))

    eval_callback = EvalCallback(test_env, best_model_save_path=exp_dir,
                                log_path=exp_dir, eval_freq=500,
                                deterministic=True, render=False)

    #model.set_parameters('%s/best_model.zip' % exp_dir)
    #evaluate_policy(model.policy, test_env, 100)

    #model.set_parameters('%s/best_model.zip' % pretrain_dir)
    model.learn(25000000, callback=eval_callback)

