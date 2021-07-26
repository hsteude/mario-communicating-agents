#!/bin/zsh
source ~/.zshrc
conda activate mario

RUNS=100
NUM_IMGS=10


for i in $(seq 1 $RUNS)
        do
                ipython ./scripts/experiment.py -- --run $i --num_imgs $NUM_IMGS
        done
