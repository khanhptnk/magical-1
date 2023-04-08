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
        next_best_rew = {
            'train_val': 0,
            'val': 0,
            'test': 0
        }
        max_videos = 5
        save_video_ids = defaultdict(set)

        for i, train_batch in zip(range(config.train.n_iters + 1), train_dataset):

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
                        save_video_every = len(dataset[split]) // max_videos

                        for j, eval_batch in enumerate(eval_dataset):
                            if len(save_video_ids[split]) < max_videos:
                                save_video_ids[split].add(eval_batch[0]['id'])
                            eval_ep_stats = self.eval_episode(
                                j, policy, eval_env, eval_batch, save_video_ids[split])
                            self._update_stats_dict(eval_stats, eval_ep_stats)
                            for (_, id), frames in eval_ep_stats['video_frames'].items():
                                for k, v in frames.items():
                                    video_name = 'example_%s_%s_%s' % (split, id, k)
                                    wandb_stats[video_name] = wandb.Video(np.array(v), fps=4, format='gif')

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

                        leap = config.train.save_every_leap
                        if leap and avg_rew > next_best_rew[split]:
                            while next_best_rew[split] < avg_rew:
                                next_best_rew[split] += leap
                            if not config.eval_mode:
                                rew_str = str('%.2f' % avg_rew).replace('.', '_')
                                policy.save('%s/new_leap_%s_%s.ckpt' % (config.exp_dir, split, rew_str))


                        wandb_stats[split + '_best_rew'] = best_rew[split]

                    policy.save('%s/last.ckpt' % config.exp_dir)
                    policy.save('%s/model_%s_%s.ckpt' % (config.exp_dir, split, i))

                if config.use_wandb:
                    wandb.log(wandb_stats)

            if config.eval_mode:
                break

            train_ep_stats = self.train_episode(i, policy, train_env, train_batch)
            self._update_stats_dict(train_stats, train_ep_stats)

    def eval_episode(self, i_iter, policy, env, batch, save_video_ids):

        def _add_frame(i, frames):
            frame = env.env_method('render', mode='rgb_array', indices=[i])[0]
            frames['allo'].append(utils.hwc_to_chw(frame['allo']))
            frames['robot_' + view].append(utils.hwc_to_chw(frame[view]))

        # set initial states in batch to environments
        for i, item in enumerate(batch):
            env.env_method('set_init_state', item['states'][0], indices=[i])
        ob = env.reset()
        policy.reset(is_eval=True)

        batch_size = len(batch)
        has_done = [False] * batch_size
        total_reward = [0] * batch_size
        num_steps = [0] * batch_size

        video_frames = {}
        for i, item in enumerate(batch):
            if item['id'] in save_video_ids:
                video_frames[(i, item['id'])] = defaultdict(list)
        view = 'allo' if 'A' in self.config.env.resolution else 'ego'

        for (i, _), v in video_frames.items():
            _add_frame(i, v)

        cnt = 0
        while not all(has_done):
            action, _ = policy.predict(ob, deterministic=True)

            ob, reward, done, info = env.step(action)

            for (i, _), v in video_frames.items():
                _add_frame(i, v)

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


