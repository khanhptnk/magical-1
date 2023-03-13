import os
import sys
import logging
import gym

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import flags
import utils
import magical
magical.register_envs()

from magical.dataset import Dataset
from magical.models import MAGICALNet, MAGICALCNN
from magical.policies import MagicalPolicy
from magical.algorithms import BehaviorCloning


if __name__ == '__main__':

    config_file, more_flags = flags.parse()
    config = utils.setup(config_file, flags=more_flags)

    env_id = '%s-Demo-%s%s-v0' % \
        (config.env.name, config.env.resolution, config.env.view)

    """
    eval_env = DummyVecEnv(
        [lambda: gym.make(env_id, config=config) for _ in range(1)])
    train_env = DummyVecEnv(
        [lambda: gym.make(env_id, config=config) for _ in range(config.train.batch_size)])
    """
    eval_env = SubprocVecEnv(
        [utils.make_env(env_id, i, config) for i in range(config.train.batch_size)])
    train_env = SubprocVecEnv(
        [utils.make_env(env_id, i, config) for i in range(config.train.batch_size)])

    policy = MagicalPolicy(
        eval_env.observation_space,
        eval_env.action_space,
        lambda x: config.policy.lr,
        net_arch=config.policy.net_arch,
        features_extractor_class=MAGICALCNN,
        features_extractor_kwargs={},
    ).to(config.device)

    if config.policy.load_from is not None:
        policy.load(config.policy.load_from)

    dataset = Dataset(config.data_dir, seed=config.seed)

    algorithm = BehaviorCloning(config, eval_env)
    algorithm.train(policy, dataset, train_env, eval_env)




