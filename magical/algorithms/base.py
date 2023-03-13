import os
import sys
import logging

from collections import defaultdict
import numpy as np
import torch


class BaseAlgorithm:

    def __init__(self, config):
        self.config = config
        self.device = config.device

    def train(self, policy, dataset, train_env, eval_env):

        config = self.config

        train_stats = defaultdict(list)
        train_dataset = dataset['train'].iterate_batches(
            cycle=True, batch_size=config.train.batch_size)

        best_rew = {
            'train_val': -1e9,
            'val': -1e9,
            'test': -1e9
        }

        for i, train_batch in zip(range(config.train.n_iters + 1), train_dataset):

            print(i)

            if i % config.train.log_every == 0:
                logging.info('Train stats %s' % ' '.join(['%s %.2f' % (k, np.average(v))
                    for k, v in train_stats.items()]))
                train_stats = defaultdict(list)

                with torch.no_grad():
                    logging.info('After %d iters' % i)

                    for split in ['train_val', 'val', 'test']:
                        eval_stats = defaultdict(list)
                        eval_dataset = dataset[split].iterate_batches(batch_size=config.train.batch_size)

                        for j, eval_batch in enumerate(eval_dataset):
                            eval_ep_stats = self.eval_episode(
                                j, policy, eval_env, eval_batch)
                            self._update_stats_dict(eval_stats, eval_ep_stats)

                        avg_rew = np.average(eval_stats['reward'])
                        avg_steps = np.average(eval_stats['num_steps'])
                        logging.info('   eval on %s, rew = %.2f, steps = %.1f' %
                            (split, avg_rew, avg_steps))

                        if split in best_rew and avg_rew > best_rew[split]:
                            best_rew[split] = avg_rew
                            if not config.eval_mode:
                                policy.save('%s/best_%s.ckpt' % (config.exp_dir, split))

                        policy.save('%s/last.ckpt' % config.exp_dir)

            if config.eval_mode:
                break

            train_ep_stats = self.train_episode(i, policy, train_env, train_batch)
            self._update_stats_dict(train_stats, train_ep_stats)

    def eval_episode(self, i_iter, policy, env, batch):
        batch_size = len(batch)
        for i, item in enumerate(batch):
            env.env_method('set_init_state', item['init_state'], indices=[i])
        ob = env.reset()
        policy.reset(is_eval=True)

        has_done = [False] * batch_size
        total_reward = [0] * batch_size
        num_steps = [0] * batch_size

        while not all(has_done):
            action, _ = policy.predict(ob, deterministic=True)
            ob, reward, done, info = env.step(action)
            for i, (r, d) in enumerate(zip(reward, done)):
                total_reward[i] += r
                if not has_done[i]:
                    num_steps[i] += 1
                has_done[i] |= d

        return dict(reward=total_reward,
                    num_steps=num_steps)

    def _update_stats_dict(self, stats, more_stats):
        for k, v in more_stats.items():
            if isinstance(v, list):
                stats[k].extend(v)
            else:
                stats[k].append(v)


