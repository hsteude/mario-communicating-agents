#!/bin/zsh
source ~/.zshrc
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_hidden_dim=200\
        --enc_rnn_num_layers=1 \
        --enc_num_hidden_states=1 \
        --learning_rate=0.001 \
        --batch_size=32 \
        --dl_num_workers=8 \
        --validdation_split=0.005 \
        --max_epochs=10 \
        --logger=True \
        --gpus=0 \
        --enc_fixed_cnn_weight \
        --enc_pretrained 
