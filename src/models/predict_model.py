import os
import torch

from models.resnet import get_config
from utils.utils import get_project_root
from models.resnet import batch_to_device, gem
from models.resnet import Net, CustomDataset
from models.common import SCORED_BIRDS

import pandas as pd


import tqdm

import argparse
import ast
from pathlib import Path

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


SCORED_BIRDS = SCORED_BIRDS()

def get_state_dict(sd_fp):
    sd = torch.load(sd_fp, map_location="cpu")["model"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    return sd


def flatten(l):
    return [item for sublist in l for item in sublist]


def create_df_test_from_path():
    files = sorted(os.listdir(TEST_AUDIO_PATH))
    data = []
    for f in files:
        wv, sr = librosa.load(TEST_AUDIO_PATH + f)
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


cfg = get_config()

PROJECT_ROOT = get_project_root()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "first_model"

state_dict = os.path.join(
    PROJECT_ROOT, "src", "models", model_name, "checkpoint_last_seed123.pth"
)
# state_dict = "../input/mel-gem-resnet-from-2021-2nd-place/first_model/checkpoint_last_seed123.pth"

if not os.path.exists(state_dict):
    raise FileNotFoundError(f"File {state_dict} not found!")


TEST_AUDIO_PATH = os.path.join(PROJECT_ROOT, "data", "interim", "test_soundscapes")



test_df = create_df_test_from_path()

N_CORES = 4
cfg.batch_size = 1
aug = None

test_ds = CustomDataset(test_df, cfg, aug, mode="val")
test_dl = DataLoader(
    test_ds,
    shuffle=False,
    batch_size=cfg.batch_size,
    num_workers=N_CORES,
)


net = Net(cfg).eval().to(DEVICE)


sd = get_state_dict(state_dict)

net.load_state_dict(sd, strict=True)

list(net.global_pool.parameters())


with torch.no_grad():
    preds = []
    for batch in tqdm(test_dl):
        batch = batch_to_device(batch, DEVICE)
        with torch.cuda.amp.autocast():
            out = net(batch)["logits"]
            preds += [out.cpu().numpy()]


df_preds = pd.DataFrame(np.vstack(preds), columns=test_ds.bird2id.keys())[SCORED_BIRDS]


test_df = test_df.join(df_preds).drop(["birds"], axis=1).reset_index()
test_df = pd.melt(
    test_df,
    id_vars=["filename", "row_prefix", "ending_second"],
    value_vars=SCORED_BIRDS,
    var_name="bird",
    value_name="proba",
)

test_df["row_id"] = (
    test_df["row_prefix"]
    + "_"
    + test_df["bird"]
    + "_"
    + test_df["ending_second"].astype(str)
)

test_df['target'] = test_df['proba'] > 0.012


sub = test_df[['row_id', 'target']]
sub.to_csv("submission.csv", index=False)


breakpoint()
