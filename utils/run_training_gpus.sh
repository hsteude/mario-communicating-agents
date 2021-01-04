#!/bin/zsh
source ~/.zshrc
#conda activate com-agent
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_hidden_dim=32\
        --enc_rnn_num_layers=2 \
        --learning_rate=0.001 \
        --batch_size=32 \
        --dl_num_workers=32 \
        --validdation_split=0.05 \
        --max_epochs=1000 \
        --logger=True \
        --gpus=4 \
        --accelerator=ddp \
        --no-enc_fixed_cnn_weight \
        --enc_pretrained 
