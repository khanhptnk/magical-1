import os
import sys
import logging
import wandb
from collections import defaultdict

import numpy as np
import torch

import utils


class BaseAlgorithm:

    def __init__(self, config):
        self.config = config
        self.device = config.device

    def train(self, policy, dataset, train_env, eval_env):

        config = self.config

        train_dataset = dataset['train'].iterate_batches(
            cycle=True, batch_size=config.train.batch_size)

        train_stats = defaultdict(list)
        wandb_stats = {}
        best_rew = {
            'train_val': -1e9,
            'val': -1e9,
            'test': -1e9
        }

        for i, train_batch in zip(range(config.train.n_iters + 1), train_dataset):

            print(i)

            if i % config.train.log_every == 0:

                wandb_stats['iter'] = i

                log_str = []
                for k, v in train_stats.items():
                    name = 'train_' + k
                    wandb_stats[name] = np.average(v)
                    log_str.append('%s %.2f' % (k, wandb_stats[name]))
                logging.info('Train stats %s' % ' '.join(log_str))
                # reset train stats
                train_stats = defaultdict(list)

                with torch.no_grad():
                    logging.info('After %d iters' % i)

                    if config.train.eval_split is not None:
                        eval_splits = [config.train.eval_splits]
                    else:
                        eval_splits = ['train_val', 'val', 'test']

                    for split in eval_splits:
                        eval_stats = defaultdict(list)
                        eval_dataset = dataset[split].iterate_batches(batch_size=config.train.batch_size)

                        for j, eval_batch in enumerate(eval_dataset):
                            save_video = (j == 0)
                            eval_ep_stats = self.eval_episode(
                                j, policy, eval_env, eval_batch, save_video=save_video)
                            self._update_stats_dict(eval_stats, eval_ep_stats)
                            if save_video:
                                for k, v in eval_ep_stats['video_frames'].items():
                                    wandb_stats['example_' + k] = wandb.Video(np.array(v), fps=4, format='gif')

                        avg_rew = np.average(eval_stats['reward'])
                        avg_steps = np.average(eval_stats['num_steps'])
                        logging.info('   eval on %s, rew = %.2f, steps = %.1f' %
                            (split, avg_rew, avg_steps))

                        wandb_stats[split + '_avg_rew'] = avg_rew
                        wandb_stats[split + '_avg_steps'] = avg_steps

                        if avg_rew > best_rew[split]:
                            best_rew[split] = avg_rew
                            if not config.eval_mode:
                                policy.save('%s/best_%s.ckpt' % (config.exp_dir, split))

                        wandb_stats[split + '_best_rew'] = best_rew[split]

                    policy.save('%s/last.ckpt' % config.exp_dir)

                wandb.log(wandb_stats)

            if config.eval_mode:
                break

            train_ep_stats = self.train_episode(i, policy, train_env, train_batch)
            self._update_stats_dict(train_stats, train_ep_stats)

    def eval_episode(self, i_iter, policy, env, batch, save_video=False):
        for i, item in enumerate(batch):
            env.env_method('set_init_state', item['init_state'], indices=[i])
        ob = env.reset()
        policy.reset(is_eval=True)

        batch_size = len(batch)
        has_done = [False] * batch_size
        total_reward = [0] * batch_size
        num_steps = [0] * batch_size

        if save_video:
            view = 'allo' if self.config.env.view == 'A' else 'ego'
            frame = env.env_method('render', mode='rgb_array', indices=[0])[0]
            video_frames = {}
            video_frames['allo'] = [utils.hwc_to_chw(frame['allo'])]
            video_frames['robot_' + view] = [utils.hwc_to_chw(frame[view])]
        else:
            video_frames = None

        cnt = 0
        while not all(has_done):
            action, _ = policy.predict(ob, deterministic=True)
            ob, reward, done, info = env.step(action)

            if save_video:
                frame = env.env_method('render', mode='rgb_array', indices=[0])[0]
                video_frames['allo'].append(utils.hwc_to_chw(frame['allo']))
                video_frames['robot_' + view].append(utils.hwc_to_chw(frame[view]))

            for i, (r, d) in enumerate(zip(reward, done)):
                total_reward[i] += r
                if not has_done[i]:
                    num_steps[i] += 1
                has_done[i] |= d
            cnt += 1

        return dict(reward=total_reward,
                    num_steps=num_steps,
                    video_frames=video_frames)

    def _update_stats_dict(self, stats, more_stats):
        for k, v in more_stats.items():
            if isinstance(v, list):
                stats[k].extend(v)
            else:
                stats[k].append(v)


