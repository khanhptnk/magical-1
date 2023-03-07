import math
import tempfile
import os
import sys
from collections import deque

import torch
import torch.nn as nn
import numpy as np
import pymunk as pm

import gym
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import make_proba_distribution
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper


import magical
magical.register_envs()
from magical.entities import RobotAction as RA


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


class SimplePolicy(BasePolicy):

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor_class,
        features_extractor_class_kwargs,
        expert,
        activation_fn=nn.ReLU):

        super().__init__(
                observation_space,
                action_space,
                features_extractor_class=features_extractor_class)

        self.net_arch = net_arch
        self.features_extractor_class = features_extractor_class
        self.features_extractor_class_kwargs = features_extractor_class_kwargs
        self.features_extractor = features_extractor_class(
            observation_space, **features_extractor_class_kwargs)

        self.action_dist = make_proba_distribution(action_space)
        self.action_net = self.action_dist.proba_distribution_net(
            latent_dim=self.features_extractor.features_dim)
        self.expert = expert

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                net_arch=self.net_arch,
                features_extractor_class=self.features_extractor_class
            )
        )
        return data

    def forward(self, obs):
        features = self.extract_features(obs)
        logits = self.action_net(features)
        return logits

    def _get_distribution(self, logits):
        return torch.distributions.Categorical(logits=logits)

    def _predict(self, obs, deterministic=False):
        logits = self.forward(obs)
        if deterministic:
            actions = logits.max(dim=-1)[1]
        actions = self._get_distribution(logits).sample()
        return actions

    #def predict(self, obs, state=None, episode_start=None, deterministic=False):
    #    return self.expert.predict(obs, deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        logits = self.forward(obs)
        distributions = self._get_distribution(logits)
        log_probs = distributions.log_prob(actions)
        entropies = distributions.entropy()
        return None, log_probs, entropies


class Expert(BasePolicy):

    def __init__(self, observation_space, action_space, features_extractor_class, venv):

        super().__init__(observation_space, action_space, features_extractor_class)
        self.envs = venv.envs
        self.history = []

    def _predict(self, states, deterministic=False):

        env = self.envs[0]

        shape = env.target_shape.shape_body

        if env.episode_steps == 0:
            self.history = deque(maxlen=10)
            self.should_hold = False

        robot_body = env.robot.robot_body
        robot_pos = robot_body.position
        robot_angle = robot_body.angle

        robot_vector = robot_body.rotation_vector.rotated_degrees(90)

        target_pos = env.target_shape.shape_body.position
        target_dist = target_pos.get_distance(robot_pos)
        target_vector = (target_pos - robot_pos).normalized()
        diff_angle = robot_vector.get_angle_between(target_vector)

        if not self.should_hold and target_dist < 0.4 and len(self.history) >= 3:
            l = len(self.history)
            for i in range(l - 3, l - 1):
                if abs(self.history[i]['dist_to_target'] - self.history[i + 1]['dist_to_target']) > 0.01:
                    self.should_hold = True

        act_flags = [RA.NONE, RA.NONE, RA.OPEN]

        if self.should_hold:
            act_flags[2] = RA.CLOSE
            target_pos = pm.vec2d.Vec2d((-1., 1.))
            target_dist = target_pos.get_distance(env.target_shape.shape_body.position)
            target_vector = (target_pos - robot_pos).normalized()
            diff_angle = robot_vector.get_angle_between(target_vector)

        angle_eps = math.pi / 20

        if abs(diff_angle) < angle_eps:
            act_flags[0] = RA.UP
        elif diff_angle < 0 and diff_angle >= -math.pi:
            act_flags[1] = RA.RIGHT
        else:
            act_flags[1] = RA.LEFT

        cnt_down = 0
        i = len(self.history) - 1
        while i >= 0 and self.history[i]['action'][0] == RA.DOWN:
            cnt_down += 1
            i -= 1

        should_back_up = False

        # continue back up
        if cnt_down > 0 and cnt_down < 10:
            should_back_up = True

        if self.history:
            print(self.history[-1]['robot_pos'], target_dist)

        cnt_unmoved = 0
        i = len(self.history) - 1
        while i > 0:
            p1 = self.history[i]['robot_pos']
            p2 = self.history[i - 1]['robot_pos']
            dp = p1.get_distance(p2)
            a1 = self.history[i]['robot_angle']
            a2 = self.history[i - 1]['robot_angle']
            da = abs(a1 - a2)
            if dp > 0.01 or da > 0.01:
                break
            cnt_unmoved += 1
            i -= 1

        if target_dist >= 0.3 and cnt_unmoved >= 3:
            should_back_up = True

        if should_back_up:
            act_flags[0] = RA.DOWN
            origin = pm.vec2d.Vec2d((0, 0))
            target_vector = (origin - robot_pos).normalized()
            diff_angle = (-robot_vector).get_angle_between(target_vector)
            if diff_angle < 0 and diff_angle >= -math.pi:
                act_flags[1] = RA.RIGHT
            else:
                act_flags[1] = RA.LEFT


        self.history.append({
            'action': act_flags,
            'robot_pos': robot_pos,
            'robot_angle': robot_angle,
            'dist_to_target': target_dist
        })

        action = env.flags_to_action(act_flags)

        input()

        return torch.tensor([action])


rng = np.random.default_rng(0)
env = gym.make('MoveToCorner-Demo-LoRes4A-v0',
                rand_shape_type=True,
                rand_shape_colour=True,
                rand_poses=True,
                debug_reward=False)
venv = DummyVecEnv([lambda: gym.make('MoveToCorner-Demo-LoRes4A-v0',
                                     rand_shape_type=True,
                                     rand_shape_colour=True,
                                     rand_poses=True,
                                     debug_reward=False)])
venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

#sys.exit(1)

expert = Expert(env.observation_space, env.action_space, MAGICALNet, venv)

policy = SimplePolicy(env.observation_space, env.action_space, [], MAGICALNet, {}, expert)

rollouts = rollout.rollout(
    expert,
    venv,
    rollout.make_sample_until(min_timesteps=None, min_episodes=100),
    rng=rng,
)

print('skkdk')

transitions = rollout.flatten_trajectories(rollouts)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
    policy=policy,
    optimizer_kwargs={ 'lr': 1e-4 },
    ent_weight=0.,
)

for _ in range(1000):
    bc_trainer.train(n_epochs=100)
    print('eval...')
    reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print("Reward:", reward)


"""
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
    policy=policy
)

with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    for i in range(1000):
        print('train')
        dagger_trainer.train(100)
        for e in venv.envs:
            e.close()
        print('eval')
        reward, _ = evaluate_policy(dagger_trainer.policy, env, 5)
        env.close()
        print(i, "Reward:", reward)
"""
