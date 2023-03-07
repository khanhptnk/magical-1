import


import magical.experts as expert_factory
from magical.algorithms import BaseAlgorithm


class BehaviorCloning(BaseAlgorithm):

    def __init__(self, config):

        super().__init__(config)
        self.expert = expert_factory.load(config)
        self.loss_fn = torch.nn.CrossEntropyLoss(
            ignore_index=-1, reduction='sum')

    def _add_ref_actions(self, ref_actions, action, done):
        for i, a in enumerate(action):
            if not done[i]:
                ref_actions[i].append(a)
            else:
                ref_actions[i].append(-1)

    def train_episode(self, i_iter, policy, env, batch):

        batch_size = len(batch)
        ob = env.reset()
        self.expert.reset(env)
        policy.reset(eval=False)

        has_done = [False] * batch_size
        total_reward = [0] * batch_size
        num_steps = [0] * batch_size

        ref_actions = [[] for _ in range(batch_size)]

        while not all(has_done):
            ref_action = self.expert()
            self._add_ref_actions(ref_actions, ref_action, has_done)
            action = policy.predict(ob, deterministic=True)
            ob, reward, done, info = env.step(ref_action)

            for i, (r, d) in enumerate(zip(reward, done)):
                total_reward[i] += r
                if not has_done[i]:
                    num_steps[i] += 1
                has_done[i] |= d

        loss = self.update_policy(policy, ref_actions)

        return dict(loss=loss,
                    rewards=total_reward,
                    num_steps=num_steps)

    def update_policy(self, policy, ref_actions):

        logits = policy.logits
        ref_actions = torch.tensor(ref_actions).to(logits.device).long()

        assert logits.shape[1] == ref_actions.shape[1]

        loss = 0
        for l, a in zip(logits, ref_actions):
            loss += self.loss_fn(l, a)

        loss /= ref_actions.shape[0]

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item() / len(ref_actions.shape[1])













