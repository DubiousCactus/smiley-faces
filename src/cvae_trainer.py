#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
CVAE trainer.
"""

from typing import List, Tuple, Union

import torch

from src.base_trainer import BaseTrainer
from utils import to_cuda
from utils.training import VAE_loss, visualize_model_predictions


class CVAE_trainer(BaseTrainer):
    def __init__(self, run_name, model, opt, train_loader, val_loader, scheduler):
        super().__init__(run_name, model, opt, train_loader, val_loader, scheduler)

    def _train_val_iteration(
        self, batch: Union[Tuple, List, torch.Tensor]
    ) -> torch.Tensor:
        """
        Train or validate model for one iteration.
        """
        x, y = to_cuda(batch)  # type: ignore
        y_pred, mu, logvar = self._model(x, y)
        return VAE_loss(y_pred, x, mu, logvar)

    def _visualize(self, batch):
        visualize_model_predictions(
            self._model, to_cuda(batch), cond=True
        )  # User implementation goes here (utils/training.py)
