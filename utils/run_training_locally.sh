#!/bin/zsh
source ~/.zshrc
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_num_layers=1 \
        --learning_rate=0.001 \
        --batch_size=32 \
        --dl_num_workers=12 \
        --validdation_split=0.005 \
        --max_epochs=1 \
        --logger=True \
        --enc_fixed_cnn_weight \
        --enc_pretrained 



