import torch

import magical.experts as expert_factory
from magical.algorithms.base import BaseAlgorithm


class BehaviorCloning(BaseAlgorithm):

    def __init__(self, config, env):

        super().__init__(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        self.expert = expert_factory.load(config,
                                          env.observation_space,
                                          env.action_space,
                                          None)

    def _add_ref_actions(self, actions, action, done):
        new_action = action[:]
        for i, a in enumerate(action):
            if done[i]:
                new_action[i] = -1
        actions.append(torch.tensor(new_action).to(self.device))

    def train_episode(self, i_iter, policy, env, batch):

        batch_size = len(batch)
        ob = env.reset()
        self.expert.reset(env)
        policy.reset(is_eval=False)

        has_done = [False] * batch_size
        total_reward = [0] * batch_size
        num_steps = [0] * batch_size

        ref_actions = []
        logits = []

        while not all(has_done):
            ref_action = self.expert.predict(ob)
            self._add_ref_actions(ref_actions, ref_action, has_done)
            action, logit = policy.forward(ob, deterministic=True)
            logits.append(logit)
            ob, reward, done, info = env.step(ref_action)

            for i, (r, d) in enumerate(zip(reward, done)):
                total_reward[i] += r
                if not has_done[i]:
                    num_steps[i] += 1
                has_done[i] |= d

        loss = self.update_policy(policy, logits, ref_actions)

        return dict(loss=loss,
                    rewards=total_reward,
                    num_steps=num_steps)

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













