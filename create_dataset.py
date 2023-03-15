import os
import sys
import json
import gym

import jsonargparse
import pickle
import flags
import utils

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import magical
magical.register_envs()
import magical.experts as expert_factory

SPLITS = ['train', 'val', 'test']

def create_set(name, env, expert, n_points):
    points = []
    batch_size = env.num_envs
    assert n_points % batch_size == 0
    id = 0
    for i in range(n_points // batch_size):
        print(name, i * batch_size)
        obs = env.reset()
        expert.reset(env)
        init_states = env.env_method('get_state', indices=range(batch_size))
        has_dones = [False] * batch_size
        action_seqs = [[] for _ in range(batch_size)]
        reward_seqs = [[] for _ in range(batch_size)]
        while not all(has_dones):
            actions = expert.predict(obs)
            obs, rewards, dones, info = env.step(actions)
            for i, (a, r, d) in enumerate(zip(actions, rewards, dones)):
                action_seqs[i].append(a)
                reward_seqs[i].append(r)
                has_dones[i] |= d
        print(rewards)
        for s, a_seq, r_seq in zip(init_states, action_seqs, reward_seqs):
            points.append(dict(id='%s_%d' % (name, id),
                               init_state=s,
                               actions=a_seq,
                               rewards=r_seq))
            id += 1

    return points

def save_all(data):
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for split in SPLITS:
        path = '%s/%s.json' % (data_dir, split)
        with open(path, 'wb') as f:
            #json.dump(data[split], f)
            pickle.dump(data[split], f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s data with %d examples to %s' % (split, len(data[split]), path))

if __name__ == '__main__':

    config_file, more_flags = flags.parse()
    config = utils.make_config(config_file, more_flags)

    env_id = '%s-TestAllButDynamics-%s-v0' % (config.env.name, config.env.resolution)

    env = SubprocVecEnv([utils.make_env(env_id, i, config) for i in range(config.train.batch_size)])
    expert = expert_factory.load(config, env.observation_space, env.action_space, None)
    create_set_fn = lambda n_points, name: create_set(name, env, expert, n_points)

    data = {}
    data['train'] = create_set_fn(config.dataset.n_train, 'train')
    data['val']   = create_set_fn(config.dataset.n_eval, 'val')
    data['test']  = create_set_fn(config.dataset.n_eval, 'test')

    save_all(data)
