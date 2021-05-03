from src.model.lit_module import LitModule
import pytorch_lightning as pl
import argparse
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from src.data.data_module import VideoDataModule
from pytorch_lightning.plugins import DDPPlugin

seed_everything(42)
# LAST_CKP = 'lightning_logs/version_16/checkpoints/epoch=11-step=3176.ckpt'


def main(args):
    # debugging forward pass
    lit_module = LitModule(**vars(args))
    # lit_module = LitModule.load_from_checkpoint(LAST_CKP, **vars(args))

    trainer = pl.Trainer.from_argparse_args(args, plugins=DDPPlugin(find_unused_parameters=False))
    # trainer = pl.Trainer(resume_from_checkpoint=LAST_CKP)
    vdm = VideoDataModule(**vars(args))
    trainer.fit(lit_module, vdm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # trainer related args
    parser = pl.Trainer.add_argparse_args(parser)

    # encoder related args
    parser.add_argument('--enc_dr_rate', type=float, default=0)
    parser.add_argument('--enc_rnn_hidden_dim', type=int, default=6)
    parser.add_argument('--enc_rnn_num_layers', type=int, default=1)
    parser.add_argument('--enc_dim_lat_space', type=int, default=5)

    # decoder related args
    parser.add_argument('--dec_num_question_inputs', type=int, default=0)
    parser.add_argument('--dec_hidden_size', type=int, default=10)
    parser.add_argument('--dec_num_hidden_layers', type=int, default=2)
    parser.add_argument('--dec_out_dim', type=int, default=6)

    # filter related args
    parser.add_argument('--filt_initial_log_var', type=float, default=0)
    parser.add_argument('--filt_num_decoders', type=int, default=3)

    # lit_module related args
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--dl_num_workers', type=int, default=12)
    parser.add_argument('--validdation_split', type=float, default=0.05)
    parser.add_argument('--pretrain_thres', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.001)
    args = parser.parse_args()
    main(args)
