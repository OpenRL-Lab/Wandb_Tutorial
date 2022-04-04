import sys
import os

import socket
from pathlib import Path
import argparse
import random

import wandb

import numpy as np
from matplotlib import pyplot as plt

def parse(args):
    parser = argparse.ArgumentParser(
        description='wandb_usage', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team_name", type=str)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--scenario_name", type=str)
    parser.add_argument("--seed",type=int,default=0)
    all_args = parser.parse_known_args(args)[0]
    return all_args

def test_matplot(args):

    all_args = parse(args)

    random.seed(all_args.seed)

    run_dir = Path("../results") / all_args.project_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    wandb.init(config=all_args,
               project=all_args.project_name,
               entity=all_args.team_name,
               notes=socket.gethostname(),
               name=all_args.experiment_name+"_"+str(all_args.seed),
               group=all_args.scenario_name,
               dir=str(run_dir),
               job_type="training",
               reinit=True)

    x = np.arange(1, 11)
    for step in range(4):
        frames = []
        y = step * x + step
        plt.title("Matplotlib Demo")
        plt.xlabel("x axis caption")
        plt.ylabel("y axis caption")
        plt.plot(x, y)
        wandb.log({"plt":wandb.Plotly(plt.gcf())},step=step)

if __name__ == '__main__':
    test_matplot(sys.argv[1:])