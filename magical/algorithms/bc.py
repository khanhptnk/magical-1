import numpy as np
import torch

import magical.experts as expert_factory
from magical.algorithms.base import BaseAlgorithm


class BehaviorCloning(BaseAlgorithm):

    def __init__(self, config, env):

        super().__init__(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

    def train_episode(self, i_iter, policy, not_used_env, batch):
        ob_iter = []
        ref_action_iter = []
        for i, item in enumerate(batch):
            ob_iter.append(iter(item['observations']))
            ref_action_iter.append(iter(item['actions']))
        policy.reset(is_eval=False)

        batch_size = len(batch)

        ref_actions = []
        logits = []

        n_steps = len(batch[0]['actions'])
        for _ in range(n_steps):
            ob = np.stack([next(o) for o in ob_iter])
            ref_action = [next(x) for x in ref_action_iter]
            ref_actions.append(torch.tensor(ref_action).to(self.device))
            action, logit = policy.forward(ob, deterministic=True)
            logits.append(logit)

        loss = self.update_policy(policy, logits, ref_actions)

        return dict(loss=loss)

    def update_policy(self, policy, logits, ref_actions):

        assert len(logits) == len(ref_actions)
        assert ref_actions[0].shape[0] == self.config.train.batch_size

        loss = 0
        for l, a in zip(logits, ref_actions):
            loss += self.loss_fn(l, a)

        loss /= self.config.train.batch_size

        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()

        return loss.item() / len(ref_actions)













