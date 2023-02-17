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
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 2 * latent_dim),  # mu, logvar
        )
        self._latent_dim = latent_dim
        self._decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim + self._condition_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
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
            self._decoder(torch.concat([z, condition], dim=-1)).view(x.shape),
            mu,
            logvar,
        )

    def sample(self, num_samples: int, c: torch.Tensor) -> torch.Tensor:
        z = torch.randn(
            num_samples, self._latent_dim, device=self._decoder[0].weight.device
        )
        return self._decoder(
            torch.concat([z, c.reshape(num_samples, self._condition_dim)], dim=1)
        ).view(num_samples, self._img_shape[0], self._img_shape[1], self._img_shape[2])


def make_seq(input_shape, output_shape, n_layers, batch_norm=True, relu=True):
    assert n_layers % 2 == 1, "must be odd fucker"
    dilation = (output_shape[1] - input_shape[1]) // (n_layers) // 2
    remainder = (output_shape[1] - input_shape[1]) % (2 * (n_layers)) // 2
    return [
        torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                input_shape[0] // 2**i,
                input_shape[0] // 2 ** (i + 1)
                if i < (n_layers - 1)
                else output_shape[0],
                3,
                dilation=dilation + (remainder if i == 0 else 0),
            ),
            torch.nn.BatchNorm2d(
                input_shape[0] // 2 ** (i + 1)
                if i < (n_layers - 1)
                else output_shape[0],
            )
            if batch_norm and i < (n_layers - 1)
            else torch.nn.Identity(),
            torch.nn.ReLU() if relu and i < (n_layers - 1) else torch.nn.Identity(),
        )
        for i in range(n_layers)
    ]


class ConvCVAE(torch.nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # C, H, W
        condition_shape: Union[Tuple[int], int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self._img_shape = image_shape
        self._condition_dim = (
            reduce(lambda a, b: a * b, condition_shape)
            if isinstance(condition_shape, tuple)
            else condition_shape
        )
        self._encoder = torch.nn.Sequential(
            torch.nn.Conv2d(image_shape[0], 32, 7),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 18),
            torch.nn.Flatten(),
            torch.nn.BatchNorm1d(256),  # More efficient
            torch.nn.ReLU(),
        )
        self._latent_encoder = torch.nn.Linear(
            256 + self._condition_dim, 2 * latent_dim
        )  # mu, logvar
        self._latent_dim = latent_dim
        self._decoder = torch.nn.Sequential(
            # *make_seq((latent_dim//4, 2, 2), image_shape, 5)
            torch.nn.ConvTranspose2d(latent_dim, 64, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 16, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 8, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 4, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(4, 2, 3),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(2, 1, 6),
        )
        # self.l1 = torch.nn.ConvTranspose2d(latent_dim, 64, 3)
        # self.l2 = torch.nn.ConvTranspose2d(64, 64, 3)
        # self.l3 = torch.nn.ConvTranspose2d(64, 64, 3)
        # self.l4 = torch.nn.ConvTranspose2d(64, 32, 3)
        # self.l5 = torch.nn.ConvTranspose2d(32, 32, 3)
        # self.l6 = torch.nn.ConvTranspose2d(32, 16, 3)
        # self.l7 = torch.nn.ConvTranspose2d(16, 16, 3)
        # self.l8 = torch.nn.ConvTranspose2d(16, 8, 3)
        # self.l9 = torch.nn.ConvTranspose2d(8, 8, 3)
        # self.l10 = torch.nn.ConvTranspose2d(8, 4, 3)
        # self.l11 = torch.nn.ConvTranspose2d(4, 2, 3)
        # self.l12 = torch.nn.ConvTranspose2d(2, 1, 6)

        print(self._encoder)
        print(self._decoder)

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        img_shape = reduce(lambda a, b: a * b, x.shape[1:])
        condition = c.reshape(-1, self._condition_dim)
        latent = self._encoder(x.view(-1, *self._img_shape))
        z_param = self._latent_encoder(torch.concat([latent, condition], dim=-1))
        mu, logvar = z_param[:, : self._latent_dim], z_param[:, self._latent_dim :]
        # 0.5 * log(var) = sqrt(var)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)
        # dist = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
        # x = self.l1(z.view(-1, self._latent_dim, 1, 1))
        # print(x.shape)
        # x = self.l2(x)
        # print(x.shape)
        # x = self.l3(x)
        # print(x.shape)
        # x = self.l4(x)
        # print(x.shape)
        # x = self.l5(x)
        # print(x.shape)
        # x = self.l6(x)
        # print(x.shape)
        # x = self.l7(x)
        # print(x.shape)
        # x = self.l8(x)
        # print(x.shape)
        # x = self.l9(x)
        # print(x.shape)
        # x = self.l10(x)
        # print(x.shape)
        # x = self.l11(x)
        # print(x.shape)
        # x = self.l12(x)
        # print(x.shape)

        return (
            self._decoder(z.view(-1, self._latent_dim, 1, 1)).view(x.shape),
            mu,
            logvar,
        )

    def sample(self, num_samples: int, c: torch.Tensor) -> torch.Tensor:
        z = torch.randn(
            num_samples, self._latent_dim, device=self._decoder[0].weight.device
        )
        return self._decoder(z.view(-1, self._latent_dim, 1, 1)).view(
            num_samples, self._img_shape[0], self._img_shape[1], self._img_shape[2]
        )
