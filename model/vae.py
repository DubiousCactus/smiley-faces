#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
VAE model.
"""


import math
from functools import reduce
from typing import Tuple, Union

import torch


class VAE(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self._img_dim = int(math.sqrt(input_dim))
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * latent_dim),  # mu, logvar
        )
        self._latent_dim = latent_dim
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_dim),
        )

    def forward(self, x):
        # batch_size = x.shape[0]
        img_shape = x.shape
        z_param = self._encoder(x.reshape(-1, img_shape[1] ** 2))
        mu, logvar = z_param[:, : self._latent_dim], z_param[:, self._latent_dim :]
        # 0.5 * log(var) = sqrt(var)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        # dist = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
        return self._decoder(z).view(img_shape), mu, logvar

    def sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(
            num_samples, self._latent_dim, device=self._decoder[0].weight.device
        )
        return self._decoder(z).view(num_samples, 1, self._img_dim, self._img_dim)


class CVAE(torch.nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # C, H, W
        condition_shape: Union[Tuple[int], int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self._img_shape = image_shape
        image_dim = reduce(lambda a, b: a * b, image_shape)
        self._condition_dim = (
            reduce(lambda a, b: a * b, condition_shape)
            if isinstance(condition_shape, tuple)
            else condition_shape
        )
        self._encoder = torch.nn.Sequential(
            torch.nn.Linear(image_dim + self._condition_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * latent_dim),  # mu, logvar
        )
        self._latent_dim = latent_dim
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + self._condition_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, image_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        img_shape = reduce(lambda a, b: a * b, x.shape[1:])
        condition = c.reshape(-1, self._condition_dim)
        z_param = self._encoder(
            torch.concat([x.reshape(-1, img_shape), condition], dim=-1)
        )
        mu, logvar = z_param[:, : self._latent_dim], z_param[:, self._latent_dim :]
        # 0.5 * log(var) = sqrt(var)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        # dist = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
        return (
            self._decoder(torch.concat([z, condition], dim=-1)).view(
                -1, *list(x.shape[1:])
            ),
            mu,
            logvar,
        )

    def sample(self, num_samples: int, c: torch.Tensor) -> torch.Tensor:
        z = torch.randn(
            num_samples, self._latent_dim, device=self._decoder[0].weight.device
        )
        return self._decoder(
            torch.concat([z, c.reshape(-1, self._condition_dim)])
        ).view(num_samples, 1, self._img_dim, self._img_dim)
