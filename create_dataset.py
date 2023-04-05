import os
import sys
import json
import gym

import numpy as np
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
    total_reward = []
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
        total_reward.extend(rewards.tolist())
        print(rewards, np.average(total_reward))
        for s, a_seq, r_seq in zip(init_states, action_seqs, reward_seqs):
            points.append(dict(id='%s_%d' % (name, id),
                               init_state=s,
                               actions=a_seq,
                               rewards=r_seq))
            id += 1

    print(np.average(total_reward))

    return points

def save_all(data):
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for split in SPLITS:
        path = '%s/%s.json' % (data_dir, split)
        with open(path, 'wb') as f:
            pickle.dump(data[split], f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s data with %d examples to %s' % (split, len(data[split]), path))

if __name__ == '__main__':

    config_file, more_flags = flags.parse()
    config = utils.make_config(config_file, more_flags)
    print(config)

    env_id = {}
    splits = ['train', 'val', 'test']

    for split in splits:
        if split in ['train', 'val']:
            env_id[split] = '%s-%s-%s-v0' % \
                (config.env.name, config.env.train_cond, config.env.resolution)
        else:
            env_id[split] = '%s-%s-%s-v0' % \
                (config.env.name, config.env.eval_cond, config.env.resolution)

    print('train env', env_id['train'])
    assert env_id['train'] == env_id['val']
    print('eval env', env_id['test'])

    env = {}
    seed_offset = 0
    for split in splits:
        env[split] = SubprocVecEnv([
            utils.make_env(env_id[split], seed_offset + i, config)
                for i in range(config.train.batch_size)])
        seed_offset += config.train.batch_size

    expert = expert_factory.load(config,
                                 env['train'].observation_space,
                                 env['train'].action_space,
                                 None)

    data = {}
    data['train'] = create_set('train', env['train'], expert, config.dataset.n_train)
    data['val']   = create_set('val', env['val'], expert, config.dataset.n_eval)
    data['test']  = create_set('test', env['test'], expert, config.dataset.n_eval)

    save_all(data)
