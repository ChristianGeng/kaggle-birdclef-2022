# -*- coding: utf-8 -*-
import shutil
import pandas as pd
import audiofile as af
import audata

import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
from utils.utils import get_project_root
import subprocess

PROJECT_ROOT = get_project_root()
DATASET_NAME = "birdclef-2022"


@click.command()
def download_data():

    target_dir = os.path.join(PROJECT_ROOT, "data", "raw")
    command = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        DATASET_NAME,
        "--path",
        target_dir,
    ]

    subprocess.run(command)


@click.command()
def unpack_data():

    fname = os.path.join(PROJECT_ROOT, "data", "raw", DATASET_NAME + ".zip")
    extract_dir = os.path.join(PROJECT_ROOT, "data", "interim")
    shutil.unpack_archive(fname, extract_dir=extract_dir)


@click.command()
def create_durations():
    """Runs data processing scripts to add"""

    # dir_name = "birdclef-2022-df-train-with-durations"
    ifname = os.path.join(get_project_root(), "data", "interim", "train_metadata.csv")
    df1 = pd.read_csv(ifname)

    audio_dur_params = [
        (os.path.join(PROJECT_ROOT, "data", "interim", "train_audio", f),)
        for f in df1["filename"]
    ]

    durations = audata.utils.run_worker_threads(
        num_workers=10,
        task_fun=af.duration,
        params=audio_dur_params,
        task_description="{:16}".format("calc durations"),
        progress_bar=True,
    )

    df = pd.concat([df1["filename"], pd.Series(durations, name='duration')], axis=1)
    ofname = os.path.join(PROJECT_ROOT, 'data', 'interim', "df-with-durations.csv")
    df.to_csv(ofname)
    logger = logging.getLogger(__name__)
    logger.info("writing duration dataset")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
