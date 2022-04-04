import sys
import os

import socket
from pathlib import Path
import argparse
import random
import math

import wandb

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

def test_curves(args):

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

    total_step_num = 1000
    for step in range(total_step_num):
        wandb.log({'random_curve':step/100+random.random()},step=step)
        wandb.log({'log_curve': math.log(step+1)},step=step)
    wandb.finish()

if __name__ == '__main__':
    test_curves(sys.argv[1:])