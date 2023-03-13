import torch

import magical.experts as expert_factory
from magical.algorithms.base import BaseAlgorithm


class BehaviorCloning(BaseAlgorithm):

    def __init__(self, config, env):

        super().__init__(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')
        """
        self.expert = expert_factory.load(config,
                                          env.observation_space,
                                          env.action_space,
                                          None)
        """

    def _add_ref_actions(self, actions, action, done):
        new_action = [-1 if done[i] else a for i, a in enumerate(action)]
        actions.append(torch.tensor(new_action).to(self.device))

    def train_episode(self, i_iter, policy, env, batch):
        batch_size = len(batch)
        ref_action_iter = []
        for i, item in enumerate(batch):
            # set initial state for environments before reset()
            env.env_method('set_init_state', item['init_state'], indices=[i])
            ref_action_iter.append(iter(item['actions']))
        ob = env.reset()
        #self.expert.reset(env)
        policy.reset(is_eval=False)

        """"
        print(batch[0]['init_state'].robot)
        print(batch[0]['init_state'].shape)
        env.env_method('render', mode='human', indices=[0])
        input()
        """

        has_done = [False] * batch_size
        total_reward = [0] * batch_size
        num_steps = [0] * batch_size

        ref_actions = []
        logits = []

        while not all(has_done):
            #ref_action = self.expert.predict(ob)
            ref_action = [next(x) for x in ref_action_iter]
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

        print(total_reward)

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













