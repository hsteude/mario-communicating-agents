#!/bin/zsh
source ~/.zshrc
conda activate mario
python ./scripts/model_training.py \
        --enc_dr_rate=0 \
        --enc_rnn_hidden_dim=32 \
        --enc_rnn_num_layers=2 \
        --num_hidden_states=3 \
        --dec_num_question_inputs=1 \
        --dec_hidden_size=32 \
        --dec_num_hidden_layers=10 \
        --dec_single_answer_dim=1 \
        --filt_initial_log_var=-10 \
        --beta=0.0 \
        --learning_rate=0.001 \
        --batch_size=32 \
        --dl_num_workers=8 \
        --validdation_split=0.05 \
        --max_epochs=1000 \
        --logger=True \
        --gpus=1 \
        --accelerator=ddp \
        --no-enc_fixed_cnn_weight \
        --enc_pretrained 
