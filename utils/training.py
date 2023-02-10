#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training utilities. This is a good place for your code that is used in training (i.e. custom loss
function, visualization code, etc.)
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def VAE_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """
    VAE loss function.
    """
    return (
        F.mse_loss(y_pred, y_true, reduction="sum")
        - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    ) / y_true.shape[0]


def visualize_model_predictions(model: torch.nn.Module, batch, cond=False) -> None:
    """
    Visualize model predictions on a dataset.
    """
    model.eval()
    with torch.no_grad():
        if not cond:
            samples = model.sample(10)  # type: ignore
        else:
            _, y = batch
            samples = model.sample(10, y[:10])  # type: ignore
        # Plot a row of 5 images with matplotlib
        fig, axs = plt.subplots(1, 10)
        for i in range(10):
            axs[i].imshow(samples[i, 0, :, :].cpu().numpy())
            axs[i].axis("off")
            if cond:
                # Plot the label under each image:
                axs[i].set_title(f"{y[:10][i].item()}")
        plt.show()
