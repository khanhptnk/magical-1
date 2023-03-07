import os
import sys
import gym

from stable_baselines3.common.vec_env import DummyVecEnv

import flags
import utils
import magical
magical.register_envs()

from magical.dataset import dataset
from magical.models import MAGICALNet
from magical.policies import MagicalPolicy
#from magical.algorithms import BehaviorCloning


if __name__ == '__main__':

    config_file, more_flags = flags.parse()
    config = utils.setup(config_file, flags=more_flags)

    env_id = '%s-TestAllButDynamics-%s%s-v0' % \
        (config.env.name, config.env.resolution, config.env.view)

    eval_env = DummyVecEnv(
        [lambda: gym.make(env_id, config=config) for _ in range(1)])
    train_env = DummyVecEnv(
        [lambda: gym.make(env_id, config=config) for _ in range(32)])

    dataset = Dataset(config.data_dir, seed=config.seed)

    """
    policy = MagicalPolicy(
        env.observation_space,
        env.action_space,
        lambda x: config.train.lr,
        net_arch=dict(pi=config.policy.pi_net_arch),
        features_extractor_class=MAGICALNet,
        features_extractor_kwargs=dict(image_dim=config.policy.image_feat_dim),
    )

    algorithm = BehaviorCloning(config)
    algorithm.train(policy, dataset, train_env, eval_env)
    """




