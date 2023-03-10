#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Smileys dataset.
"""


import os
from typing import Optional, Tuple, Union

import torch

from dataset.base.image import ImageDataset


class SmileysDataset(ImageDataset):
    IMG_DIM = (2160, 3840)

    def __init__(
        self,
        dataset_root: str,
        split: str,
        img_dim: Optional[int] = None,
        augment=False,
        normalize=False,
        tiny=False,
        identity: Optional[Union[str, list]] = None,
    ) -> None:
        self._identity = identity
        super().__init__(dataset_root, split, img_dim, augment, normalize, tiny)

    def _load(
        self, dataset_root: str, tiny: bool, split: Optional[str] = None
    ) -> Tuple[Union[dict, list, torch.Tensor], Union[dict, list, torch.Tensor]]:
        samples, labels = [], []
        if self._identity is not None and not isinstance(self._identity, list):
            self._identity = [self._identity]
        for root, _, files in os.walk(dataset_root):
            if self._identity is not None:
                if os.path.basename(root) not in self._identity:
                    continue
            for file in files:
                if file.endswith(".jpg"):
                    samples.append(os.path.join(root, file))
                    labels.append(torch.tensor(float(file.split(".")[0])))
        # TODO: split here
        return samples, labels
