# -*- coding: utf-8 -*-
import shutil
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
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def df_with_durations():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """

    dir_name = "birdclef-2022-df-train-with-durations"
    ofname = "df-with-durations.csv"

    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
