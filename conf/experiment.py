#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Configurations for the experiments and config groups, using hydra-zen.
"""

from dataclasses import dataclass
from test import launch_test
from typing import Optional

import hydra_zen
import torch
from hydra.conf import HydraConf, JobConf, RunDir
from hydra_zen import ZenStore, builds, make_config, make_custom_builds_fn, store
from torch.utils.data import DataLoader
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from dataset.celeba import CelebADataset
from dataset.mnist import MNISTDataset
from model.vae import CVAE, VAE, ConvCVAE
from src.base_trainer import BaseTrainer
from src.cvae_trainer import CVAE_trainer
from train import launch_experiment

# Set hydra.job.chdir=True using store():
hydra_store = ZenStore(overwrite_ok=True)
hydra_store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra")
# We'll generate a unique name for the experiment and use it as the run name
hydra_store(
    HydraConf(
        run=RunDir(
            f"runs/{get_random_name(combo=[ADJECTIVES, NAMES], separator='-', style='lowercase')}"
        )
    ),
    name="config",
    group="hydra",
)
hydra_store.add_to_hydra_store()
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=False)

" ================== Dataset ================== "
# Dataclasses are a great and simple way to define a base config group with default values.
@dataclass
class ImageDatasetConf:
    tiny: bool = False
    normalize: bool = False
    augment: bool = False


# Pre-set the group for store's dataset entries
dataset_store = store(group="dataset")
dataset_store(
    pbuilds(
        MNISTDataset,
        builds_bases=(ImageDatasetConf,),
        dataset_root="data/mnist",
        img_dim=MNISTDataset.IMG_SIZE[0],
    ),
    name="mnist",
)
dataset_store(
    pbuilds(
        CelebADataset,
        builds_bases=(ImageDatasetConf,),
        dataset_root="data/celeba",
        img_dim=CelebADataset.IMG_SIZE[0],
    ),
    name="celeba",
)

" ================== Dataloader & sampler ================== "


@dataclass
class SamplerConf:
    batch_size: int = 16
    drop_last: bool = True
    shuffle: bool = True


@dataclass
class DataloaderConf:
    batch_size: int = 128
    drop_last: bool = True
    shuffle: bool = True


" ================== Model ================== "
# Pre-set the group for store's model entries
model_store = store(group="model")

# Not that encoder_input_dim depend on dataset.img_dim, so we need to use a partial to set them in
# the launch_experiment function.
model_store(
    pbuilds(
        VAE,
        latent_dim=128,
        image_shape=hydra_zen.MISSING,
        use_convolutional_encoder=True,
    ),
    name="vae",
)

model_store(
    pbuilds(
        CVAE,
        latent_dim=128,
        condition_shape=1,
        image_shape=hydra_zen.MISSING,
        use_convolutional_encoder=True,
    ),
    name="cvae",
)
model_store(
    pbuilds(ConvCVAE, latent_dim=128, condition_shape=1, image_shape=hydra_zen.MISSING),
    name="conv_cvae",
)

" ================== Optimizer ================== "


@dataclass
class Optimizer:
    lr: float = 1e-3
    weight_decay: float = 0.0


opt_store = store(group="optimizer")
opt_store(
    pbuilds(
        torch.optim.Adam,
        builds_bases=(Optimizer,),
    ),
    name="adam",
)
opt_store(
    pbuilds(
        torch.optim.SGD,
        builds_bases=(Optimizer,),
    ),
    name="sgd",
)


" ================== Scheduler ================== "
sched_store = store(group="scheduler")
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.StepLR,
        step_size=100,
        gamma=0.5,
    ),
    name="step",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        mode="min",
        factor=0.5,
        patience=10,
    ),
    name="plateau",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        T_max=100,
    ),
    name="cosine",
)

" ================== Experiment ================== "


@dataclass
class TrainingConfig:
    epochs: int = 200
    seed: int = 42
    val_every: int = 1
    viz_every: int = 10
    load_from_path: Optional[str] = None
    load_from_run: Optional[str] = None


training_store = store(group="training")
training_store(TrainingConfig, name="default")

trainer_store = store(group="trainer")
trainer_store(
    pbuilds(
        BaseTrainer,
    ),
    name="base",
)

trainer_store(
    pbuilds(
        CVAE_trainer,
        # specialized_argument=specialized_argument,
    ),
    name="cvae_trainer",
)

Experiment = builds(
    launch_experiment,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"training": "default"},
        {"dataset": "mnist"},
        {"model": "vae"},
        {"optimizer": "adam"},
        {"scheduler": "cosine"},
        {"trainer": "base"},
    ],
    data_loader=pbuilds(
        DataLoader, builds_bases=(DataloaderConf,)
    ),  # Needs a partial because we need to set the dataset
)
store(Experiment, name="base_experiment")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment", package="_global_")
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /dataset": "mnist"},
            {"override /model": "vae"},
        ],
        model=dict(image_shape=(1, 28, 28)),
        bases=(Experiment,),
    ),
    name="vae_mnist",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /dataset": "mnist"},
            {"override /model": "cvae"},
            {"override /trainer": "cvae_trainer"},
        ],
        model=dict(image_shape=(1, 28, 28)),
        bases=(Experiment,),
    ),
    name="cvae_mnist",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /dataset": "celeba"},
            {"override /model": "vae"},
        ],
        model=dict(image_shape=(3, 64, 64)),
        bases=(Experiment,),
    ),
    name="vae_celeba",
)
" ================== Model testing ================== "


@dataclass
class TestingConfig:
    seed: int = 42
    viz_every: int = 10
    load_from_path: Optional[str] = None
    load_from_run: Optional[str] = None


training_store = store(group="testing")
training_store(TestingConfig, name="default")


ExperimentEvaluation = builds(
    launch_test,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"dataset": "image_a"},
        {"model": "model_a"},
        {"testing": "default"},
    ],
    data_loader=pbuilds(
        DataLoader, builds_bases=(DataloaderConf,)
    ),  # Needs a partial because we need to set the dataset
)
store(ExperimentEvaluation, name="base_experiment_evaluation")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment_evaluation", package="_global_")
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "model_a"},
            {"override /dataset": "image_a"},
        ],
        bases=(ExperimentEvaluation,),
    ),
    name="exp_a",
)
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "model_b"},
            {"override /dataset": "image_b"},
        ],
        bases=(ExperimentEvaluation,),
    ),
    name="exp_b",
)
