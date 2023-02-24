#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
VAE model.
"""


from functools import reduce
from typing import List, Tuple, Union

import torch


class MLP(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, layers: List[int], batchnorm: bool = True
    ) -> None:
        super().__init__()
        _layers = [torch.nn.Linear(input_dim, layers[0]), torch.nn.ReLU()]
        if batchnorm:
            _layers.append(torch.nn.BatchNorm1d(layers[0]))
        for i in range(1, len(layers)):
            _layers += [torch.nn.Linear(layers[i - 1], layers[i]), torch.nn.ReLU()]
            if batchnorm:
                _layers.append(torch.nn.BatchNorm1d(layers[i]))
        _layers.append(torch.nn.Linear(layers[-1], output_dim))
        self._layers = torch.nn.Sequential(*_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


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


class CNN(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        output_dim: int,
        pooling: bool = False,
    ) -> None:
        super().__init__()
        self._layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[0], 32, 7),
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
            torch.nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(self._layers(x).shape)
        return self._layers(x)


class CNNDecoder(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int], output_shape: Tuple[int]) -> None:
        super().__init__()
        self._layers = torch.nn.Sequential(
            # *make_seq((latent_dim//4, 2, 2), image_shape, 5)
            torch.nn.ConvTranspose2d(input_shape[0], 64, 3),
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
            torch.nn.ConvTranspose2d(2, output_shape[0], 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)


def reparameterization_trick(
    z_param: torch.Tensor, latent_dim: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, logvar = z_param[:, :latent_dim], z_param[:, latent_dim:]
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar), mu, logvar


class VAE(torch.nn.Module):
    def __init__(
        self,
        image_shape: Tuple[int, int, int],  # C, H, W
        latent_dim: int,
        convolutional_encoder: bool = True,
    ) -> None:
        super().__init__()
        self._img_shape = image_shape
        image_dim = reduce(lambda a, b: a * b, image_shape)
        self._latent_dim = latent_dim
        self._encoder = (
            MLP(image_dim, 2 * latent_dim, [512, 256], batchnorm=True)
            if not convolutional_encoder
            else CNN(image_shape, 2 * latent_dim)
        )
        self._decoder = MLP(latent_dim, image_dim, [256, 512], batchnorm=True)
        self._use_conv = convolutional_encoder

    def forward(self, x: torch.Tensor):
        input_dim = reduce(lambda a, b: a * b, x.shape[1:])
        # print(self._encoder)
        # print(input_dim, x.shape)
        z_param = self._encoder(x if self._use_conv else x.reshape(-1, input_dim))
        # print(z_param.shape)
        z, mu, logvar = reparameterization_trick(z_param, self._latent_dim)
        return self._decoder(z).view(x.shape), mu, logvar

    def sample(self, num_samples: int) -> torch.Tensor:
        z = torch.randn(
            num_samples,
            self._latent_dim,
            device="cuda:0",  # self._decoder.weight.device
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
        self._encoder = MLP(
            image_dim + self._condition_dim, 2 * latent_dim, [512, 512, 256]
        )
        self._latent_dim = latent_dim
        self._decoder = MLP(
            latent_dim + self._condition_dim, image_dim, [256, 512, 512]
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor):
        img_shape = reduce(lambda a, b: a * b, x.shape[1:])
        condition = c.reshape(-1, self._condition_dim)
        z_param = self._encoder(
            torch.concat([x.reshape(-1, img_shape), condition], dim=-1)
        )
        z, mu, logvar = reparameterization_trick(z_param, self._latent_dim)
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
            torch.nn.ConvTranspose2d(2, 3, 6),
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
        print(latent)
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
