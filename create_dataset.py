import os
import sys
import json
import gym

import orjson
import jsonargparse
import jsonpickle
import pickle
import flags
import utils

import magical
magical.register_envs()
import magical.experts as expert_factory

SPLITS = ['train', 'val', 'test']

def create_set(name, env, expert, n_points):
    points = []
    for i in range(n_points):
        print(i)
        ob = env.reset()
        expert.reset([env])
        init_state = env.get_state()
        obs = [ob]
        actions = []
        has_done = False
        rewards = []
        while not has_done:
            action = expert.predict(ob)[0]
            ob, reward, done, info = env.step(action)
            obs.append(ob)
            actions.append(action)
            rewards.append(reward)
            has_done |= done
        points.append(dict(id='%s_%d' % (name, i),
                           init_state=init_state,
                           actions=actions,
                           rewards=rewards))
    return points

def save_all(data):
    data_dir = config.data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for split in SPLITS:
        path = '%s/%s.json' % (data_dir, split)
        with open(path, 'w') as f:
            json.dump(data[split], f)
            #pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('Saved %s data with %d examples to %s' % (split, len(data[split]), path))

if __name__ == '__main__':

    config_file, more_flags = flags.parse()
    config = utils.make_config(config_file, more_flags)

    env_id = '%s-TestAllButDynamics-%s%s-v0' % \
        (config.env.name, config.env.resolution, config.env.view)

    env = gym.make(env_id, config=config)

    expert = expert_factory.load(config,
                                 env.observation_space,
                                 env.action_space,
                                 None)

    create_set_fn = lambda n_points, name: create_set(name, env, expert, n_points)

    data = {}
    data['train'] = create_set_fn(config.dataset.n_train, 'train')
    data['val']   = create_set_fn(config.dataset.n_eval, 'val')
    data['test']  = create_set_fn(config.dataset.n_eval, 'test')

    save_all(data)
