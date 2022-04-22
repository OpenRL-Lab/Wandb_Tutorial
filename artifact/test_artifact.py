import os
import sys
import argparse
from pathlib import Path
import time

import torch
import wandb

from load_data import load,read,preprocess

def parse(args):
    parser = argparse.ArgumentParser(
        description='wandb_usage', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team_name", type=str)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--scenario_name", type=str)
    parser.add_argument("--wandb_log_path", type=str, default="../../wandb_results/")
    parser.add_argument("--seed",type=int,default=0)
    all_args = parser.parse_known_args(args)[0]
    return all_args

def set_wandb(all_args):
    run_dir = Path("../../../wandb_results") / all_args.project_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    os.environ["WANDB_ENTITY"] = all_args.team_name
    os.environ["WANDB_PROJECT"] = all_args.project_name
    os.environ["WANDB_DIR"] = str(run_dir)
    return run_dir

def test_upload_artifact(run):
    datasets = load()  # separate code for loading the datasets
    names = ["training"]

    # 🏺 create our Artifact
    raw_data = wandb.Artifact(
        "mnist-origin", type="dataset",
        description="Raw MNIST dataset, split into train/val/test",
        metadata={"source": "torchvision.datasets.MNIST",
                  "sizes": [len(dataset) for dataset in datasets]})

    for name, data in zip(names, datasets):
        # 🐣 Store a new file in the artifact, and write something into its contents.
        with raw_data.new_file(name + ".pt", mode="wb") as file:
            x, y = data.tensors
            torch.save((x, y), file)

    # ✍️ Save the artifact to W&B.
    run.log_artifact(raw_data)

def test_download_artifact(run):

    # 对数据进行预处理后，再进行保存
    steps = {"normalize": True,
             "expand_dims": True}
    processed_data = wandb.Artifact(
        "mnist-preprocess", type="dataset",
        description="Preprocessed MNIST dataset",
        metadata=steps)

    # ✔️ declare which artifact we'll be using
    raw_data_artifact = run.use_artifact('mnist-origin:latest')

    # 📥 if need be, download the artifact
    raw_dataset = raw_data_artifact.download()

    for split in ["training"]:
        raw_split = read(raw_dataset, split)
        processed_dataset = preprocess(raw_split, **steps)

        with processed_data.new_file(split + ".pt", mode="wb") as file:
            x, y = processed_dataset.tensors
            torch.save((x, y), file)

    run.log_artifact(processed_data)

def count_down(duration:int):
    for remaining in range(duration, 0, -1):
        sys.stdout.write("\r")
        sys.stdout.write("{:2d} seconds remaining...".format(remaining))
        sys.stdout.flush()
        time.sleep(1)


def test_artifact(args):
    all_args = parse(args)
    run_dir = set_wandb(all_args)

    # 🚀 start a run, with a type to label it and a project it can call home
    with wandb.init(entity=all_args.team_name,
                    project=all_args.project_name,
                    config=all_args,
                    dir=str(run_dir),
                    name=all_args.experiment_name + "_" + str(all_args.seed),
                    job_type="load-data") as run:
        test_upload_artifact(run)

        count_down(10)

        test_download_artifact(run)

if __name__ == '__main__':
    test_artifact(sys.argv[1:])