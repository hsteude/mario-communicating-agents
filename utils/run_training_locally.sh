#!/bin/zsh
source ~/.zshrc
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_hidden_dim=32 \
        --enc_rnn_num_layers=1 \
        --enc_dim_lat_space=4 \
        --dec_num_question_inputs=0 \
        --dec_hidden_size=32 \
        --dec_num_hidden_layers=3 \
        --dec_out_dim=1 \
        --filt_initial_log_var=-10 \
        --filt_num_decoders=3 \
        --pretrain_thres=0.1 \
        --beta=0.001 \
        --learning_rate=0.001 \
        --batch_size=32 \
        --dl_num_workers=4 \
        --validdation_split=0.005 \
        --max_epochs=1 \
        --logger=True \
        --gpus=0 \
        --enc_pretrained \
        --enc_fixed_cnn_weight
