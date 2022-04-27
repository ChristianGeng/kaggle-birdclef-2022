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
from utils.utils import get_project_root

PROJECT_ROOT = get_project_root()

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_config():

    cfg = SimpleNamespace()

    # paths
    cfg.data_folder = ""
    cfg.name = "julian"
    # cfg.data_dir = "../input/birdclef-2022/"
    cfg.data_dir = os.path.join(get_project_root(), 'data', 'interim')

    # cfg.train_data_folder = cfg.data_dir + "train_audio/"
    cfg.train_data_folder = os.path.join(cfg.data_dir, "train_audio/")

    # cfg.val_data_folder = cfg.data_dir + "train_audio/"
    # cfg.val_data_folder = os.path.join(cfg.data_dir, "train_audio/")
    cfg.val_data_folder = os.path.join(cfg.data_dir, "test_soundscapes/")

    cfg.output_dir = "first_model"

    # dataset
    cfg.dataset = "base_ds"
    cfg.min_rating = 0
    cfg.val_df = None
    cfg.batch_size_val = 1
    cfg.train_aug = None
    cfg.val_aug = None
    cfg.test_augs = None

    cfg.wav_len_val = 5  # seconds

    # audio
    cfg.window_size = 2048
    cfg.hop_size = 512
    cfg.sample_rate = 32000
    cfg.fmin = 16
    cfg.fmax = 16386
    cfg.power = 2
    cfg.mel_bins = 256
    cfg.top_db = 80.0

    # img model
    cfg.backbone = "resnet18"
    cfg.pretrained = True
    cfg.pretrained_weights = None
    cfg.train = True
    cfg.val = False
    cfg.in_chans = 1

    cfg.alpha = 1
    cfg.eval_epochs = 1
    cfg.eval_train_epochs = 1
    cfg.warmup = 0

    cfg.mel_norm = False

    cfg.label_smoothing = 0

    cfg.remove_pretrained = []

    # training
    cfg.seed = 123
    cfg.save_val_data = True

    # ressources
    cfg.mixed_precision = True
    cfg.gpu = 0
    cfg.num_workers = 4  # 18
    cfg.drop_last = True

    cfg.mixup2 = 0

    cfg.label_smoothing = 0

    cfg.mixup_2x = False

    cfg.birds = np.array(
        [
            "afrsil1",
            "akekee",
            "akepa1",
            "akiapo",
            "akikik",
            "amewig",
            "aniani",
            "apapan",
            "arcter",
            "barpet",
            "bcnher",
            "belkin1",
            "bkbplo",
            "bknsti",
            "bkwpet",
            "blkfra",
            "blknod",
            "bongul",
            "brant",
            "brnboo",
            "brnnod",
            "brnowl",
            "brtcur",
            "bubsan",
            "buffle",
            "bulpet",
            "burpar",
            "buwtea",
            "cacgoo1",
            "calqua",
            "cangoo",
            "canvas",
            "caster1",
            "categr",
            "chbsan",
            "chemun",
            "chukar",
            "cintea",
            "comgal1",
            "commyn",
            "compea",
            "comsan",
            "comwax",
            "coopet",
            "crehon",
            "dunlin",
            "elepai",
            "ercfra",
            "eurwig",
            "fragul",
            "gadwal",
            "gamqua",
            "glwgul",
            "gnwtea",
            "golphe",
            "grbher3",
            "grefri",
            "gresca",
            "gryfra",
            "gwfgoo",
            "hawama",
            "hawcoo",
            "hawcre",
            "hawgoo",
            "hawhaw",
            "hawpet1",
            "hoomer",
            "houfin",
            "houspa",
            "hudgod",
            "iiwi",
            "incter1",
            "jabwar",
            "japqua",
            "kalphe",
            "kauama",
            "laugul",
            "layalb",
            "lcspet",
            "leasan",
            "leater1",
            "lessca",
            "lesyel",
            "lobdow",
            "lotjae",
            "madpet",
            "magpet1",
            "mallar3",
            "masboo",
            "mauala",
            "maupar",
            "merlin",
            "mitpar",
            "moudov",
            "norcar",
            "norhar2",
            "normoc",
            "norpin",
            "norsho",
            "nutman",
            "oahama",
            "omao",
            "osprey",
            "pagplo",
            "palila",
            "parjae",
            "pecsan",
            "peflov",
            "perfal",
            "pibgre",
            "pomjae",
            "puaioh",
            "reccar",
            "redava",
            "redjun",
            "redpha1",
            "refboo",
            "rempar",
            "rettro",
            "ribgul",
            "rinduc",
            "rinphe",
            "rocpig",
            "rorpar",
            "rudtur",
            "ruff",
            "saffin",
            "sander",
            "semplo",
            "sheowl",
            "shtsan",
            "skylar",
            "snogoo",
            "sooshe",
            "sooter1",
            "sopsku1",
            "sora",
            "spodov",
            "sposan",
            "towsol",
            "wantat1",
            "warwhe1",
            "wesmea",
            "wessan",
            "wetshe",
            "whfibi",
            "whiter",
            "whttro",
            "wiltur",
            "yebcar",
            "yefcan",
            "zebdov",
        ]
    )

    cfg.n_classes = len(cfg.birds)
    # dataset
    cfg.min_rating = 2.0

    cfg.wav_crop_len = 30  # seconds

    cfg.lr = 0.0001
    cfg.epochs = 5
    cfg.batch_size = 64
    cfg.batch_size_val = 64
    cfg.backbone = "resnet34"

    cfg.save_val_data = True
    cfg.mixed_precision = True

    cfg.mixup = True
    cfg.mix_beta = 1

    # cfg.train_df1 = "../input/birdclef-2022/train_metadata.csv"
    cfg.train_df1 = os.path.join(get_project_root(), 'data', 'interim',"train_metadata.csv")


    # cfg.train_df2 = "../input/birdclef-2022-df-train-with-durations/df-with-durations.csv"
    # cfg.train_df2 = "../input/birdclef-2022-df-train-with-durations/df-with-durations.csv"
    cfg.train_df2 = os.path.join(PROJECT_ROOT, 'data', 'interim', "df-with-durations.csv")


    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.tr_collate_fn = None
    cfg.val_collate_fn = None
    cfg.val = False

    cfg.dev = False

    cfg.model = "RN34"

    return cfg

cfg = get_config()

def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()

        self.bird2id = {bird: idx for idx, bird in enumerate(cfg.birds)}
        if self.mode == "train":
            self.data_folder = cfg.train_data_folder
            self.df = self.df[self.df["rating"] >= self.cfg.min_rating]
        elif self.mode == "val":
            self.data_folder = cfg.val_data_folder
        elif self.mode == "test":
            self.data_folder = cfg.test_data_folder

        self.fns = self.df["filename"].unique()

        self.df = self.setup_df()

        self.aug_audio = cfg.train_aug

    def setup_df(self):
        df = self.df.copy()

        if self.mode == "train":

            df["weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)
            df["target"] = df["primary_label"].apply(self.bird2id.get)
            labels = np.eye(self.cfg.n_classes)[df["target"].astype(int).values]
            label2 = (
                df["secondary_labels"].apply(lambda x: self.secondary2target(x)).values
            )
            for i, t in enumerate(label2):
                labels[i, t] = 1
        else:
            targets = df["birds"].apply(lambda x: self.birds2target(x)).values
            labels = np.zeros((df.shape[0], self.cfg.n_classes))
            # import pdb; pdb.set_trace()
            for i, t in enumerate(targets):
                labels[i, t] = 1

        df[[f"t{i}" for i in range(self.cfg.n_classes)]] = labels

        if self.mode != "train":
            df = df.groupby("filename")

        return df

    def __getitem__(self, idx):


        # vermeiden dass etwas nicht geht
        wav = None

        if self.mode == "train":
            row = self.df.iloc[idx]
            fn = row["filename"]
            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            weight = row["weight"]
            # fold = row["fold"]
            fold = -1

            # wav_len = row["length"]
            parts = 1
        else:
            fn = self.fns[idx]
            row = self.df.get_group(fn)
            label = row[[f"t{i}" for i in range(self.cfg.n_classes)]].values
            wav_len = None
            # Este es mi "entrada" a que un audio dure mucho
            parts = label.shape[0]
            fold = -1
            weight = 1

        if self.mode == "train":
            # wav_len_sec = wav_len / self.cfg.sample_rate
            wav_len_sec = row["duration"]
            duration = self.cfg.wav_crop_len
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        else:
            offset = 0.0
            duration = None

        wav = self.load_one(fn, offset, duration)

        if wav.shape[0] < (self.cfg.wav_crop_len * self.cfg.sample_rate):
            pad = self.cfg.wav_crop_len * self.cfg.sample_rate - wav.shape[0]
            wav = np.pad(wav, (0, pad))

        if self.mode == "train":
            if self.aug_audio:
                wav = self.aug_audio(samples=wav, sample_rate=self.cfg.sample_rate)
        else:
            if self.cfg.val_aug:
                wav = self.cfg.val_aug(samples=wav, sample_rate=self.cfg.sample_rate)

        wav_tensor = torch.tensor(wav)  # (n_samples)
        if parts > 1:
            n_samples = wav_tensor.shape[0]
            wav_tensor = wav_tensor[: n_samples // parts * parts].reshape(
                parts, n_samples // parts
            )

        feature_dict = {
            "input": wav_tensor,
            "target": torch.tensor(label.astype(np.float32)),
            "weight": torch.tensor(weight),
            "fold": torch.tensor(fold),
        }
        return feature_dict

    def __len__(self):
        if cfg.dev:
            return 256
        return len(self.fns)

    def load_one(self, id_, offset, duration):
        fp = self.data_folder + id_
        try:
            wav, sr = librosa.load(fp, sr=None, offset=offset, duration=duration)
        except:
            print("FAIL READING rec", fp)

        return wav

    def birds2target(self, birds):
        # birds = birds.split()
        target = [self.bird2id.get(item) for item in birds if not item == "nocall"]
        return target

    def secondary2target(self, secondary_label):
        birds = ast.literal_eval(secondary_label)
        target = [self.bird2id.get(item) for item in birds if not item == "nocall"]
        return target




#------------------
# Dataset related utils
#------------------

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
    train_df = pd.merge(train_df1[['primary_label', 'secondary_labels', 'rating', 'filename']], train_df2[['filename', 'duration']], how='inner', on='filename')
    return train_df


#------------------
# GeM and Mix-up
#------------------



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
            return X, Y


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()

        self.cfg = cfg

        self.n_classes = cfg.n_classes

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.window_size,
            win_length=cfg.window_size,
            hop_length=cfg.hop_size,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            pad=0,
            n_mels=cfg.mel_bins,
            power=cfg.power,
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)

        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            global_pool="",
            in_chans=cfg.in_chans,
        )

        if "efficientnet" in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = GeM()

        self.head = nn.Linear(backbone_out, self.n_classes)

        if cfg.pretrained_weights is not None:
            sd = torch.load(cfg.pretrained_weights, map_location="cpu")["model"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
            print("weights loaded from", cfg.pretrained_weights)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        self.mixup = Mixup(mix_beta=cfg.mix_beta)

        self.factor = int(cfg.wav_crop_len / 5.0)

    def forward(self, batch):

        if not self.training:
            x = batch["input"]
            bs, parts, time = x.shape
            x = x.reshape(parts, time)
            y = batch["target"]
            y = y[0]
        else:
            x = batch["input"]
            y = batch["target"]
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)

        with autocast(enabled=False):
            x = self.wav2img(x)  # (bs, mel, time)
            if self.cfg.mel_norm:
                x = (x + 80) / 80

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]

        weight = batch["weight"]

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            if self.cfg.mixup:
                # breakpoint()
                x, y, weight = self.mixup(x, y, weight)
            if self.cfg.mixup2:
                x, y, weight = self.mixup(x, y, weight)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)

        x = self.backbone(x)

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logits = self.head(x)

        loss = self.loss_fn(logits, y)
        loss = (loss.mean(dim=1) * weight) / weight.sum()
        loss = loss.sum()

        return {
            "loss": loss,
            "logits": logits.sigmoid(),
            "logits_raw": logits,
            "target": y,
        }


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


def SCORED_BIRDS():
    fname = os.path.join(get_project_root(), "data", "interim", "scored_birds.json")
    with open(fname) as fp:
        SCORED_BIRDS = json.load(fp)
    return SCORED_BIRDS


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
#TEST_AUDIO_ROOT = "../input/birdclef-2022/test_soundscapes/"
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
    test_dl = DataLoader(test_ds, shuffle=False, batch_size = cfg.batch_size, num_workers = N_CORES)
    test_ds[0]

    DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'

    # pkg_resources.resource_exists(resource_package, resource_path):
    # resource_package = "data"

    # pkg_resources.
    # resource_path = "/".join(("data", "raw","test_soundscapes"))

    # cfg.val_data_folder = TEST_AUDIO_ROOT
    # cfg.pretrained = False
    breakpoint()


def train():

    set_seed(cfg.seed)


    #val_df = pd.read_csv(cfg.val_df)
    #val_dataset = CustomDataset(val_df, cfg, aug=cfg.val_aug, mode="val")

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

            torch.save(checkpoint, f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}_{epoch}.pth")

    if cfg.epochs > 0:
        checkpoint = create_checkpoint(model, optimizer, epoch, scheduler=scheduler, scaler=scaler)

        torch.save(checkpoint, f"{cfg.output_dir}/checkpoint_last_seed{cfg.seed}.pth")



if __name__ == "__main__":
    train()
