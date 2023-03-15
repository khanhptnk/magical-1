import os
import sys
import time
import logging
import random
import yaml
from datetime import datetime
import wandb

import numpy as np
import gym
import torch

from stable_baselines3.common.utils import set_random_seed


class ElapsedFormatter():

    def __init__(self):
        self.start_time = datetime.now()

    def format_time(self, t):
        return str(t)[:-7]

    def format(self, record):
        elapsed_time = self.format_time(datetime.now() - self.start_time)
        log_str = "%s %s: %s" % (elapsed_time,
                                record.levelname,
                                record.getMessage())

        return log_str


class Config:
    def __init__(self, **entries):
        rec_entries = {}
        for k, v in entries.items():
            if isinstance(v, dict):
                rv = Config(**v)
            elif isinstance(v, list):
                rv = []
                for item in v:
                    if isinstance(item, dict):
                        rv.append(Config(**item))
                    else:
                        rv.append(item)
            else:
                rv = v
            rec_entries[k] = rv
        self.__dict__.update(rec_entries)

    def __str_helper(self, depth):
        lines = []
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                v_str = v.__str_helper(depth + 1)
                lines.append("%s:\n%s" % (k, v_str))
            else:
                lines.append("%s: %r" % (k, v))
        indented_lines = ["    " * depth + l for l in lines]
        return "\n".join(indented_lines)

    def __str__(self):
        return "Config {\n%s\n}" % self.__str_helper(1)

    def __repr__(self):
        return "Config(%r)" % self.__dict__

    def __getattr__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return None

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


def make_config(yaml_file_or_str, flags=None):
    with open(yaml_file_or_str) as f:
        config_dict = yaml.safe_load(f)

    if flags is not None:
        update_config(flags, config_dict)

    config = Config(**config_dict)

    return config


def setup(yaml_file_or_str, flags=None):

    config = make_config(yaml_file_or_str, flags=flags)

    config.exp_dir = "%s/%s" % (config.exp_root_dir, config.name)
    if not config.eval_mode:
        assert not os.path.exists(config.exp_dir), "Experiment %s already exists!" % config.exp_dir
        os.makedirs(config.exp_dir)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    config.random = random.Random(config.seed)
    config.device = torch.device('cuda:%d' % config.device)

    config.start_time = time.time()
    if config.eval_mode:
        log_file = os.path.join(config.exp_dir, 'eval.log')
    else:
        log_file = os.path.join(config.exp_dir, 'train.log')
    config_logging(log_file)
    logging.info('python -u ' + ' '.join(sys.argv))
    logging.info(str(datetime.now()))
    logging.info('Write log to %s' % log_file)
    logging.info(str(config))

    if config.use_wandb:
        wandb.init(
            project='inferlearn',
            group='magical',
            name=config.name,
            id=config.wandb_id,
            config=config.to_dict()
        )

    return config

def config_logging(log_file):

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(ElapsedFormatter())

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(ElapsedFormatter())

    logging.basicConfig(level=logging.INFO,
                        handlers=[stream_handler, file_handler])

    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))

    sys.excepthook = handler

def update_config(source, target):
    for k in source.keys():
        if isinstance(source[k], dict):
            if k not in target:
                target[k] = {}
            update_config(source[k], target[k])
        elif source[k] is not None:
            target[k] = source[k]

def make_env(env_id, rank, config):
    def _init():
        env = gym.make(env_id, config=config)
        env.seed(config.seed + rank)
        return env
    return _init

def hwc_to_chw(img):
    return img.transpose((2, 0, 1))
