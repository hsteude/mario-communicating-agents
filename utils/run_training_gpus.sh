#!/bin/zsh
source ~/.zshrc
#conda activate com-agent
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_hidden_dim=200\
        --enc_rnn_num_layers=3 \
        --learning_rate=0.0001 \
        --batch_size=32 \
        --dl_num_workers=8 \
        --validdation_split=0.005 \
        --max_epochs=100 \
        --logger=True \
        --gpus=1 \
        --no-enc_fixed_cnn_weight \
        --enc_pretrained 





