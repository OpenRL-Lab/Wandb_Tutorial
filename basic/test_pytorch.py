import sys
import os

import socket
from pathlib import Path
import argparse
import random
import math

import numpy as np
import wandb
import torch

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.hid1 = torch.nn.Linear(4, 7)  # 4-7-3
    self.oupt = torch.nn.Linear(7, 3)

  def forward(self, x):
    z = torch.tanh(self.hid1(x))
    z = self.oupt(z)
    return z


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

def test_pytorch(args):

    all_args = parse(args)

    # set all the seed
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    np.random.seed(all_args.seed)

    run_dir = Path("../results") / all_args.project_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    wandb.init(config=all_args,
               project=all_args.project_name,
               entity=all_args.team_name,
               notes=socket.gethostname(),
               name=all_args.experiment_name + "_" + str(all_args.seed),
               group=all_args.scenario_name,
               dir=str(run_dir),
               job_type="training",
               reinit=True)

    train_x = np.array([
        [5.0, 3.5, 1.3, 0.3],
        [4.5, 2.3, 1.3, 0.3],
        [5.5, 2.6, 4.4, 1.2],
        [6.1, 3.0, 4.6, 1.4],
        [6.7, 3.1, 5.6, 2.4],
        [6.9, 3.1, 5.1, 2.3]], dtype=np.float32)

    train_y = np.array([0, 0, 1, 1, 2, 2], dtype=np.long)

    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    model = Model()
    wandb.watch(model,log_freq=1)
    max_epochs = 100
    lrn_rate = 0.04
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lrn_rate)

    model.train()

    indices = np.arange(6)
    for epoch in range(0, max_epochs):
        np.random.shuffle(indices)
        for i in indices:
            X = train_x[i].reshape(1, 4)
            Y = train_y[i].reshape(1, )
            optimizer.zero_grad()
            oupt = model(X)
            loss_obj = loss_func(oupt, Y)
            loss_obj.backward()
            optimizer.step()

            print('Epoch:{} Loss:{}'.format(epoch,loss_obj))

    wandb.finish()

if __name__ == '__main__':
    test_pytorch(sys.argv[1:])