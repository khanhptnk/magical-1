import os
import sys
import logging

import torch
import torch.nn as nn

from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    MlpExtractor,
    get_actor_critic_arch,
)


class MagicalPolicy(BasePolicy):

    def __init__(self,
            observation_space,
            action_space,
            lr_schedule=None,
            net_arch=None,
            activation_fn=nn.ReLU,
            features_extractor_class=None,
            features_extractor_kwargs=None,
            normalize_images: bool = True,
            optimizer_class=torch.optim.AdamW,
            optimizer_kwargs=None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.AdamW:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim

        self.dist_kwargs = None
        self.action_dist = make_proba_distribution(
            action_space, dist_kwargs=self.dist_kwargs)

        self._build(lr_schedule)

        n_params = sum(p.numel() for p in self.parameters())
        logging.info(self)
        logging.info('Number of parameters: %d' % n_params)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        modules = create_mlp(
            self.features_dim,
            -1,
            self.net_arch,
            activation_fn=self.activation_fn,
        )
        self.mlp_extractor = nn.Sequential(*modules)

    def _build(self, lr_schedule):
        self._build_mlp_extractor()
        latent_dim_pi = self.net_arch[-1]
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def extract_features(self, obs):
        return super().extract_features(obs, self.features_extractor)

    def _get_action_dist_from_latent(self, logits):
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        obs, vectorized_env = self.obs_to_tensor(obs)
        pi_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor(pi_features)
        logits = self.action_net(latent_pi)
        distribution = self._get_action_dist_from_latent(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, logits

    def get_distribution(self, obs):
        pi_features = self.extract_features(obs)
        latent_pi = self.mlp_extractor(pi_features)
        logits = self.action_net(latent_pi)
        return self._get_action_dist_from_latent(logits)

    def _predict(self, obs, deterministic=False):
        actions = self.get_distribution(obs).get_actions(deterministic=deterministic)
        return actions

    def reset(self, is_eval=False):
        self.train(not is_eval)

    def save(self, path):
        checkpoint = {}
        checkpoint['state_dict'] = self.state_dict()
        checkpoint['optim'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        logging.info('Saved model to %s' % path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optim'])
        logging.info('Loaded model from %s' % path)
