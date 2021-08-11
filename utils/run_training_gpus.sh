#!/bin/zsh
source ~/.zshrc
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_hidden_dim=50 \
        --enc_pretrained=Ture \
        --enc_fixed_cnn_weights=False \
        --enc_rnn_num_layers=2 \
        --enc_dim_lat_space=4 \
        --dec_num_question_inputs=0 \
        --dec_hidden_size=10 \
        --dec_num_hidden_layers=2 \
        --dec_out_dim=1 \
        --filt_initial_log_var=-10 \
        --filt_num_decoders=4 \
        --pretrain_thres=0.03 \
        --beta=0.0001 \
        --learning_rate=0.001 \
        --batch_size=64 \
        --dl_num_workers=24 \
        --validdation_split=0.1 \
        --logger=True \
        --gpus=1
