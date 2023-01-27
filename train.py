#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training code.
"""

import os

import hydra_zen
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from hydra_zen import just, store, zen
from hydra_zen.typing import Partial

import conf.experiment  # Must import the config to add all components to the store!
from conf import project as project_conf
from src.base_trainer import BaseTrainer
from utils import colorize, seed_everything, to_cuda


def launch_experiment(
    training,
    dataset: torch.utils.data.Dataset,
    data_loader: Partial[torch.utils.data.DataLoader],
    model: Partial[torch.nn.Module],
    optimizer: Partial[torch.optim.Optimizer],
    scheduler: Partial[torch.optim.lr_scheduler._LRScheduler],
):
    run_name = os.path.basename(HydraConfig.get().runtime.output_dir)
    # Generate a random ANSI code:
    color_code = f"38;5;{hash(run_name) % 255}"
    print(
        colorize(
            f"========================= Running {run_name} =========================",
            color_code,
        )
    )

    "============ Partials instantiation ============"
    model_inst = model(
        encoder_input_dim=just(dataset).img_dim
    )  # Use just() to get the config out of the Zen-Partial
    train_dataset, val_dataset = dataset(split="train"), dataset(split="val")
    opt_inst = optimizer(model_inst.parameters())
    scheduler_inst = scheduler(
        opt_inst
    )  # TODO: handle the epoch parameter for CosineAnnealingLR

    "============ CUDA ============"
    model_inst: torch.nn.Module = to_cuda(model_inst)  # type: ignore

    "============ Weights & Biases ============"
    if project_conf.USE_WANDB:
        wandb.init(project=project_conf.PROJECT_NAME, name=run_name)
    " ============ Reproducibility of data loaders ============ "
    g = None
    if project_conf.REPRODUCIBLE:
        g = torch.Generator()
        g.manual_seed(training.seed)

    train_loader_inst = data_loader(train_dataset, generator=g)
    val_loader_inst = data_loader(val_dataset, generator=g)

    " ============ Training ============ "
    model_ckpt_path = None

    if training.load_from_run is not None and training.load_from_path is not None:
        raise ValueError(
            "Both training.load_from_path and training.load_from_run are set. Please choose only one."
        )
    elif training.load_from_run is not None:
        run_models = sorted(
            [
                f
                for f in os.listdir(to_absolute_path(f"runs/{training.load_from_run}/"))
                if f.endswith(".ckpt")
            ]
        )
        if len(run_models) < 1:
            raise ValueError(f"No model found in runs/{training.load_from_run}/")
        model_ckpt_path = to_absolute_path(
            os.path.join(
                "runs",
                training.load_from_run,
                run_models[-1],
            )
        )
    elif training.load_from_path is not None:
        model_ckpt_path = to_absolute_path(training.load_from_path)

    BaseTrainer(
        run_name=run_name,
        model=model_inst,
        opt=opt_inst,
        scheduler=scheduler_inst,
        train_loader=train_loader_inst,
        val_loader=val_loader_inst,
    ).train(
        epochs=training.epochs,
        val_every=training.val_every,
        visualize_every=training.viz_every,
        model_ckpt_path=model_ckpt_path,
    )


if __name__ == "__main__":
    "============ Hydra-Zen ============"
    store.add_to_hydra_store(
        overwrite_ok=True
    )  # Overwrite Hydra's default config to update it
    zen(
        launch_experiment,
        pre_call=hydra_zen.zen(
            lambda training: seed_everything(
                training.seed
            )  # training is the config of the training group, part of the base config
            if project_conf.REPRODUCIBLE
            else lambda: None
        ),
    ).hydra_main(
        config_name="base_experiment",
        version_base="1.3",  # Hydra base version
    )
