import sys
import flags
import utils
import logging
import numpy as np
import magical
import wandb

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

model = DiscreteVAE(image_size=96).to(config.device)

logging.info(model)


dataset = Dataset(config,
                  splits=['train', 'val'],
                  data_split_cls=VAEDataSplit,
                  prefix='dvae_',
                  seed=config.seed)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)

train_iter = dataset['train'].iterate_batches(batch_size=config.train.batch_size, cycle=True)

wandb_stats = {}
train_losses = []
for i, train_batch in zip(range(config.train.n_iters + 1), train_iter):

    if i % config.train.log_every == 0:

        wandb_stats['iter'] = i

        avg_train_loss = np.average(train_losses)
        logging.info('Train loss = %.4f' % avg_train_loss)
        wandb_stats['train_loss'] = avg_train_loss

        train_losses = []

        for split in ['train_val', 'val']:
            eval_iter = dataset[split].iterate_batches(
                batch_size=config.train.batch_size, cycle=False)
            model.eval()

            eval_losses = []
            recon_losses = []

            for j, eval_batch in enumerate(eval_iter):
                eval_batch = process_batch(eval_batch)
                eval_loss, output = model(eval_batch, return_loss=True, return_recons=True)
                eval_losses.append(eval_loss.item())
                recon_losses.append(F.mse_loss(eval_batch, output).item())

                if j == 0:
                    saved_image = (output[0] * 255).long()
                    saved_image[saved_image < 0] = 0
                    saved_image[saved_image > 255] = 255
                    saved_image = saved_image.permute(1, 2, 0).cpu().numpy()

                    saved_image = wandb.Image(saved_image)
                    wandb_stats['recon_image_%s' % split] = saved_image

            avg_eval_loss = np.average(eval_losses)
            avg_recon_loss = np.average(recon_losses)
            logging.info('   %s loss = %.4f   recon_loss = %.4f' %
                (split, avg_eval_loss, avg_recon_loss))
            wandb_stats['eval_loss_%s' % split] = avg_eval_loss
            wandb_stats['recon_loss_%s' % split] = avg_recon_loss

        if config.use_wandb:
            wandb.log(wandb_stats)



    model.train()
    train_batch = process_batch(train_batch)
    train_loss = model(train_batch, return_loss=True)
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    train_losses.append(train_loss.item())







