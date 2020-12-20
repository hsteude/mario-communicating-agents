#!/bin/zsh
source ~/.zshrc
conda activate com-agent

RUNS=10

for i in $(seq 1 $RUNS)
        do
                ipython ./mario_game/source/main.py -- --run $i &
        done
