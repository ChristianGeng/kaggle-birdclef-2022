import argparse
import ast
from pathlib import Path

from models.common import SCORED_BIRDS
import gc
import glob
import importlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
from collections import defaultdict
from copy import copy
from types import SimpleNamespace

from click import utils

import cv2
import librosa
import numpy as np
import pandas as pd
import pkg_resources
import timm
import torch
import torchaudio as ta
from torch import nn
from torch import optim
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.distributions import Beta
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import SequentialSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from models.resnet import get_config
from utils.utils import get_project_root
from models.resnet import batch_to_device, gem
from models.resnet import Net, CustomDataset
from models.common import SCORED_BIRDS


PROJECT_ROOT = get_project_root()


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


cfg = get_config()

# ------------------
# Dataset related utils
# ------------------


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def get_train_dataloader(train_ds, cfg):
    train_dataloader = DataLoader(
        train_ds,
        sampler=None,
        shuffle=True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=cfg.tr_collate_fn,
        drop_last=cfg.drop_last,
        worker_init_fn=worker_init_fn,
    )
    print(f"train: dataset {len(train_ds)}, dataloader {len(train_dataloader)}")
    return train_dataloader


def get_scheduler(cfg, optimizer, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps=cfg.epochs * (total_steps // cfg.batch_size),
    )
    return scheduler


def load_df(cfg):
    train_df1 = pd.read_csv(cfg.train_df1)
    train_df2 = pd.read_csv(cfg.train_df2)
    train_df = pd.merge(
        train_df1[["primary_label", "secondary_labels", "rating", "filename"]],
        train_df2[["filename", "duration"]],
        how="inner",
        on="filename",
    )
    return train_df


# ------------------
# GeM and Mix-up
# ------------------


class GeM(nn.Module):
    # Generalized mean: https://arxiv.org/abs/1711.02512
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", eps="
            + str(self.eps)
            + ")"
        )


class Mixup(nn.Module):
    def __init__(self, mix_beta):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, X, Y, weight=None):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        if weight is None:
            return X, Y
        else:
            weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
            return X, Y, weight

# zum Trainieren!
def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint


def get_state_dict(sd_fp):
    sd = torch.load(sd_fp, map_location="cpu")["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd


def create_df_test_from_path():
    files = sorted(os.listdir(TEST_AUDIO_PATH))
    data = []
    for f in files:
        # breakpoint()
        wv, sr = librosa.load(os.path.join(TEST_AUDIO_PATH, f))
        n_chunks = math.ceil(len(wv) / sr / 5)
        filename = f
        row_prefix = f[:-4]
        bird = SCORED_BIRDS[0]
        for chunk in range(1, n_chunks + 1):
            # for bird in SCORED_BIRDS:
            # row_id = f"{f[:-4]}_{bird}_{chunk*5}"

            ending_second = chunk * 5
            data.append((filename, row_prefix, ending_second, [bird]))

    return pd.DataFrame(
        data, columns=["filename", "row_prefix", "ending_second", "birds"]
    )


SCORED_BIRDS = SCORED_BIRDS()
# TEST_AUDIO_ROOT = "../input/birdclef-2022/test_soundscapes/"
TEST_AUDIO_ROOT = os.path.join(get_project_root(), "data", "raw", "test_soundscapes")

# TEST_AUDIO_PATH = '../input/birdclef-2022/test_soundscapes/'
TEST_AUDIO_PATH = os.path.join(get_project_root(), "data", "raw", "test_soundscapes")


def val_dinger():

    # print(cfg.model, cfg.dataset, cfg.backbone, cfg.pretrained_weights, cfg.mel_norm)

    test_df = create_df_test_from_path()

    N_CORES = 4
    cfg.batch_size = 1

    aug = None
    test_ds = CustomDataset(test_df, cfg, aug, mode="val")
    test_dl = DataLoader(
        test_ds, shuffle=False, batch_size=cfg.batch_size, num_workers=N_CORES
    )
    test_ds[0]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # pkg_resources.resource_exists(resource_package, resource_path):
    # resource_package = "data"

    # pkg_resources.
    # resource_path = "/".join(("data", "raw","test_soundscapes"))

    # cfg.val_data_folder = TEST_AUDIO_ROOT
    # cfg.pretrained = False
    breakpoint()


def train():

    set_seed(cfg.seed)

    # val_df = pd.read_csv(cfg.val_df)
    # val_dataset = CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")

    train_df = load_df(cfg)

    train_dataset = CustomDataset(train_df, cfg, aug=cfg.train_aug, mode="train")
    train_dataloader = get_train_dataloader(train_dataset, cfg)
    model = Net(cfg)

    model.to(cfg.device)

    total_steps = len(train_dataset)

    params = model.parameters()
    optimizer = optim.Adam(params, lr=cfg.lr, weight_decay=0)
    scheduler = get_scheduler(cfg, optimizer, total_steps)

    device = cfg.device

    try:
        os.makedirs(cfg.output_dir)
    except:
        pass

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    cfg.curr_step = 0
    i = 0
    best_val_loss = np.inf
    optimizer.zero_grad()
    for epoch in range(cfg.epochs):

        set_seed(cfg.seed + epoch)

        cfg.curr_epoch = epoch

        print("EPOCH:", epoch)

        progress_bar = tqdm(range(len(train_dataloader)))
        tr_it = iter(train_dataloader)

        losses = []

        gc.collect()

        if cfg.train:
            # ==== TRAIN LOOP
            for itr in progress_bar:
                i += 1

                cfg.curr_step += cfg.batch_size

                data = next(tr_it)

                model.train()
                torch.set_grad_enabled(True)

                batch = batch_to_device(data, device)

                if cfg.mixed_precision:
                    with autocast():
                        output_dict = model(batch)
                else:
                    output_dict = model(batch)

                loss = output_dict["loss"]

                losses.append(loss.item())

                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if cfg.curr_step % cfg.batch_size == 0:
                    progress_bar.set_description(f"loss: {np.mean(losses[-10:]):.4f}")

        if cfg.val:
            if (epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs:
                val_loss = run_eval(model, val_dataloader, cfg)
            else:
                val_score = 0

        if cfg.epochs > 0:
            checkpoint = create_checkpoint(
                model, optimizer, epoch, scheduler=scheduler, scaler=scaler
            )

            torch.save(
                checkpoint,
                f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}_{epoch}.pth",
            )

    if cfg.epochs > 0:
        checkpoint = create_checkpoint(
            model, optimizer, epoch, scheduler=scheduler, scaler=scaler
        )

        torch.save(checkpoint, f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth")


if __name__ == "__main__":
    train()
