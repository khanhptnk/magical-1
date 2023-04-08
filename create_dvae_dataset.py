import os
import sys
import json
import gym
import random
from collections import deque

import numpy as np
import jsonargparse
import pickle
import flags
import utils
import cv2

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import magical
magical.register_envs()
from magical.dataset import Dataset
from magical.policies import MagicalPolicy
from magical.models import MAGICALNet, MAGICALCNN


SPLITS = ['train', 'val']

def rollout(batch, policy, env):

    for i, item in enumerate(batch):
        env.env_method('set_init_state', item['init_state'], indices=[i])

    ob = env.reset()
    policy.reset(is_eval=True)

    batch_size = len(batch)
    has_done = [False] * batch_size
    rewards = [[] for _ in range(batch_size)]
    states = [[] for _ in range(batch_size)]
    observations = [[] for _ in range(batch_size)]
    actions = [[] for _ in range(batch_size)]
    ids = list(range(batch_size))

    state = env.env_method('get_state', indices=ids)
    view_dict = env.env_method('render', mode='rgb_array')
    for i in range(batch_size):
        img = view_dict[i]['allo']
        img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        img = img.transpose((2, 0, 1))
        observations[i].append(img)
        states.append(state[i])

    cnt = 0
    print('DEBUG!!! change deterministic to False')
    while not all(has_done):

        # sample action instead of taking argmax
        action, _ = policy.predict(ob, deterministic=True)
        ob, reward, done, info = env.step(action)

        view_dict = env.env_method('render', mode='rgb_array')

        state = env.env_method('get_state', indices=ids)
        for i in range(batch_size):
            img = view_dict[i]['allo']
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
            img = img.transpose((2, 0, 1))
            observations[i].append(img)
            states[i].append(state[i])
            actions[i].append(action[i])
            rewards[i].append(reward[i])
            has_done[i] |= done[i]

        cnt += 1

    trajs = [None] * batch_size
    for i in range(batch_size):
        trajs[i] = {
            'observations': observations[i],
            'actions': actions[i],
            'states': states[i],
            'rewards': rewards[i]
        }

    total_reward = [r[-1] for r in rewards]
    #print(total_reward)

    return trajs, total_reward

def save_all(data):
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for split in SPLITS:
        path = '%s/dvae_%s.json' % (data_dir, split)
        with open(path, 'wb') as f:
            pickle.dump(data[split], f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s data with %d examples to %s' % (split, len(data[split]), path))

def make_trajs(data_split, policy, env, max_trajs=10):
    total_reward = []
    trajs = []
    data_split.shuffle()
    data_iter = data_split.iterate_batches(
        batch_size=config.train.batch_size, cycle=False)
    trajs = []
    for batch in data_iter:
        new_trajs, new_rew = rollout(batch, policy, env)
        trajs.extend(new_trajs)
        total_reward.extend(new_rew)
        if len(trajs) >= max_trajs:
            break
    print(np.average(total_reward))
    return trajs

if __name__ == '__main__':

    config_file, more_flags = flags.parse()
    config = utils.make_config(config_file, more_flags)
    config.exp_dir = "%s/%s" % (config.exp_root_dir, config.name)

    print(config)
    dataset = Dataset(config, seed=config.seed)

    env_id = '%s-Demo-%s-v0' % \
        (config.env.name, config.env.resolution)
    env = SubprocVecEnv([utils.make_env(env_id, i, config) for i in range(config.train.batch_size)])

    policy = MagicalPolicy(
        env.observation_space,
        env.action_space,
        lambda x: config.policy.lr,
        net_arch=config.policy.net_arch,
        features_extractor_class=MAGICALCNN,
        features_extractor_kwargs={},
    ).to(config.device)

    model_iters = [0, 50, 100, 150, 300, 600, 1500, 3000]
    trajs = []
    for it in model_iters:
        model_path = '%s/model_test_%d.ckpt' % (config.exp_dir, it)
        policy.load(model_path)
        print(model_path)
        trajs.extend(make_trajs(dataset['train'], policy, env))
        trajs.extend(make_trajs(dataset['test'], policy, env))
        print(len(trajs))

    random.shuffle(trajs)
    data = {}
    data['val'] = trajs[:len(trajs) // 10]
    data['train'] = trajs[len(trajs) // 10:]

    save_all(data)
