#!/bin/zsh
source ~/.zshrc
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_hidden_dim=10 \
        --enc_rnn_num_layers=1 \
        --enc_dim_lat_space=4 \
        --dec_num_question_inputs=0 \
        --dec_hidden_size=10 \
        --dec_num_hidden_layers=2 \
        --dec_out_dim=1 \
        --filt_initial_log_var=-5 \
        --filt_num_decoders=3 \
        --pretrain_thres=0.4 \
        --beta=0.01 \
        --learning_rate=0.001 \
        --batch_size=32 \
        --dl_num_workers=12 \
        --validdation_split=0.05 \
        --logger=True \
        --gpus=1 \
        --no-enc_fixed_cnn_weight \
        --enc_pretrained \
        --accelerator=ddp
        #--max_epochs=1000 \
