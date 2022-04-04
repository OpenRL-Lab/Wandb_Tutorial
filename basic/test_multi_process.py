import sys
import os
from multiprocessing import Process

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

def running_func(args,start_step):

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

    total_step_num = 10
    for step in range(total_step_num):
        if step % 2 == start_step:
            wandb.log({'log_curve': math.log(step+1)},step=step)
    wandb.finish()

def test_multi_process(args):
    process = [Process(target=running_func, args=(args,0)),
               Process(target=running_func, args=(args,1)), ]
    [p.start() for p in process]  # 开启了两个进程
    [p.join() for p in process]  # 等待两个进程依次结束

if __name__ == '__main__':
    test_multi_process(sys.argv[1:])