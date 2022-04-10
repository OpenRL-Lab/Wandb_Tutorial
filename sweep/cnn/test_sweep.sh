#!/bin/bash

unset TMUX

python test_sweep.py --team_name "tmarl" --project_name "wandb_usage" --experiment_name "test_sweep_cnn" \
 --scenario_name "test_sweep_cnn" --sweep_worker_num 4