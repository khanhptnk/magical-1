import os
import sys
import flags
import math
import utils
import logging
import numpy as np
import magical
import wandb
import pickle

magical.register_envs()

import torch
torch.backends.cudnn.deterministic = True
import torch.nn.functional as F

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from magical.dataset import Dataset
from magical.models import DiscreteVAE
from magical.policies import MagicalPolicy
from magical.algorithms import BehaviorCloning
from magical.dataset import VAEDataSplit

def process_batch(images):
    batch = torch.tensor(np.stack(images)).to(config.device).float()
    # normalize images
    batch /= 255.0
    return batch

config_file, more_flags = flags.parse()
config = utils.setup(config_file, flags=more_flags)

model = DiscreteVAE(image_size=96,
                    num_tokens=config.dvae.num_tokens,
                    codebook_dim=config.dvae.codebook_dim,
                    hidden_dim=config.dvae.hidden_dim,
    ).to(config.device)

logging.info(model)

#exp_dir = 'experiments/move_to_corner_hard_1k_dvae_ntokens_128_codebook_256_maxkl_0'
model_path = os.path.join(config.exp_dir, 'best_val_recon_loss.ckpt')
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
logging.info('Loaded model from %s' % model_path)

dataset = Dataset(config,
                  splits=['train', 'val'],
                  data_split_cls=VAEDataSplit,
                  prefix='dvae_',
                  seed=config.seed)

split = 'val'

eval_iter = dataset[split].iterate_batches(
    batch_size=config.train.batch_size, cycle=False)
model.eval()

saved_data = []

for j, eval_batch in enumerate(eval_iter):
    eval_batch = process_batch(eval_batch)
    with torch.no_grad():
        codebook_indices = model.get_codebook_indices(eval_batch)
        codebook_indices = codebook_indices.view(codebook_indices.shape[0], -1)
        images = model.decode(codebook_indices)

    for x, y in zip(codebook_indices, images):
        saved_data.append((x, y))

print(len(saved_data))

save_path = os.path.join(config.exp_dir, 'val_codes_images.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(saved_data, f, pickle.HIGHEST_PROTOCOL)










