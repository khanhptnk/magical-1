import sys
import flags
import math
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

def cosine_schedule(t_cur, t_max, n_min, n_max):
    t_cur = min(t_cur, t_max)
    return n_min + 0.5 * (n_max - n_min) * (1 + math.cos(math.pi * t_cur / t_max))

def kl_weight_schedule(iter):
    max_weight = config.dvae.max_kl_weight
    min_weight = 0
    max_iter = config.train.n_iters // 2
    return cosine_schedule(iter, max_iter, max_weight, min_weight) / (96*96*3)

def temp_schedule(iter):
    max_temp = config.dvae.max_temp
    min_temp = 0.5
    max_iter = config.train.n_iters // 2
    return cosine_schedule(iter, max_iter, min_temp, max_temp)


config_file, more_flags = flags.parse()
config = utils.setup(config_file, flags=more_flags)

model = DiscreteVAE(image_size=96,
                    num_tokens=config.dvae.num_tokens,
                    codebook_dim=config.dvae.codebook_dim,
                    hidden_dim=config.dvae.hidden_dim,
    ).to(config.device)

logging.info(model)


dataset = Dataset(config,
                  splits=['train', 'val'],
                  data_split_cls=VAEDataSplit,
                  prefix='dvae_',
                  seed=config.seed)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, config.train.n_iters // 2, eta_min=1.25 * 1e-6)

train_iter = dataset['train'].iterate_batches(batch_size=config.train.batch_size, cycle=True)

best_metric = {
    'loss': 1e9,
    'recon_loss': 1e9
}

wandb_stats = {}
train_losses = []
for i, train_batch in zip(range(config.train.n_iters + 1), train_iter):

    if i % config.train.log_every == 0:

        wandb_stats['iter'] = i

        avg_train_loss = np.average(train_losses)
        cur_kl_weight = kl_weight_schedule(i)
        cur_temp = temp_schedule(i)
        cur_lr = optimizer.param_groups[0]['lr']
        logging.info('Iter %d, train loss = %.4f, beta = %.8f, temp = %.4f, lr = %.8f' %
            (i, avg_train_loss, cur_kl_weight, cur_temp, cur_lr))
        wandb_stats['train_loss'] = avg_train_loss
        wandb_stats['lr'] = cur_lr
        wandb_stats['kl_weight'] = cur_kl_weight
        wandb_stats['temp'] = cur_temp

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
                    recon_image = (output[0] * 255).long()
                    recon_image[recon_image < 0] = 0
                    recon_image[recon_image > 255] = 255
                    recon_image = recon_image.permute(1, 2, 0).cpu().numpy()

                    recon_image = wandb.Image(recon_image)
                    wandb_stats['recon_image_%s' % split] = recon_image

                    orign_image = eval_batch[0] * 255
                    orign_image = orign_image.permute(1, 2, 0).cpu().numpy()
                    orign_image = wandb.Image(orign_image)
                    wandb_stats['orign_image_%s' % split] = orign_image

            eval_metric = {}
            eval_metric['loss']= np.average(eval_losses)
            eval_metric['recon_loss'] = np.average(recon_losses)
            logging.info('   %s loss = %.4f   recon_loss = %.4f' %
                (split, eval_metric['loss'], eval_metric['recon_loss']))
            wandb_stats['eval_loss_%s' % split] = eval_metric['loss']
            wandb_stats['recon_loss_%s' % split] = eval_metric['recon_loss']

            if split == 'val':
                for k in eval_metric:
                    if eval_metric[k] < best_metric[k]:
                        best_metric[k] = eval_metric[k]
                        model_path = '%s/best_%s_%s.ckpt' % (config.exp_dir, split, k)
                        torch.save(model.state_dict(), model_path)
                        logging.info('Saved model to %s' % model_path)


        if config.use_wandb:
            wandb.log(wandb_stats)

    model.train()
    train_batch = process_batch(train_batch)

    train_loss = model(train_batch,
        return_loss=True,
        temp=temp_schedule(i),
        kl_div_loss_weight=kl_weight_schedule(i),
        debug=(i % 20) == 0
    )

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    lr_scheduler.step()

    train_losses.append(train_loss.item())







