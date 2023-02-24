#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
CelebA dataset.
"""


from typing import Optional, Tuple, Union

import torch
from torchvision.datasets import CelebA
from torchvision.transforms import transforms

from dataset.base.image import ImageDataset


class CelebADataset(ImageDataset):
    IMG_SIZE = (64, 64)

    def __init__(
        self,
        dataset_root: str,
        split: str,
        img_dim: Optional[int] = None,
        augment=False,
        normalize=False,
        tiny=False,
    ) -> None:
        super().__init__(dataset_root, split, img_dim, augment, normalize, tiny)
        split = "valid" if split == "val" else split
        self._dataset = CelebA(
            root=dataset_root,
            split=split,
            download=True,
            target_type="identity",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        self.IMG_SIZE[0] + 8 if img_dim is None else img_dim + 8
                    ),
                    transforms.CenterCrop(
                        self.IMG_SIZE if img_dim is None else img_dim
                    ),
                    transforms.ToTensor(),
                ]
            ),
        )

    def _load(
        self, dataset_root: str, tiny: bool, split: str
    ) -> Tuple[Union[dict, list, torch.Tensor], Union[dict, list, torch.Tensor]]:
        return [], []

    def __getitem__(self, index: int):
        sample, label = self._dataset[index]
        if self._augment:
            sample = self._augs(image=sample)["image"]
        if self._normalize:
            sample = self._normalization(sample)
        return sample, label

    def __len__(self):
        return len(self._dataset)
