#!/bin/bash

unset TMUX

python test_sweep.py --team_name "tmarl" --project_name "wandb_usage" --experiment_name "test_sweep" \
 --scenario_name "test_sweep" --sweep_worker_num 2