import os
import sys
import json
import gym

import jsonargparse
import jsonpickle
import flags
import utils

import magical
magical.register_envs()
import magical.experts as expert_factory

SPLITS = ['train', 'val', 'test']

def create_set(env, expert, n_points):
    points = []
    for i in range(n_points):
        ob = env.reset()
        expert.reset([env])
        init_state = env.get_state()
        actions = []
        has_done = False
        rewards = []
        while not has_done:
            action = expert.predict(ob)[0]
            ob, reward, done, info = env.step(action)
            actions.append(action)
            rewards.append(reward)
            has_done |= done
        points.append(dict(init_state=init_state,
                           actions=actions,
                           rewards=rewards))
    return points

def save_all(data):
    data_dir = config.data_dir
    print(data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for split in SPLITS:
        path = '%s/%s.json' % (data_dir, split)
        print('Saving %s data with %d examples to %s' % (split, len(data[split]), path))
        with open(path, 'w') as f:
            json.dump(data, f)

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

    create_set_fn = lambda n_points: create_set(env, expert, n_points)

    data = {}
    data['train'] = create_set_fn(config.dataset.n_train)
    data['val']   = create_set_fn(config.dataset.n_eval)
    data['test']  = create_set_fn(config.dataset.n_eval)

    save_all(data)
