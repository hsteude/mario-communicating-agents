from src.model.lit_module_formula_only import LitModule
import pytorch_lightning as pl
import argparse
# from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

seed_everything(1)
LAST_CKP = 'lightning_logs/version_8/checkpoints/epoch=179-step=1791.ckpt'
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main(args):
    # debugging forward pass
    # lit_module = LitModule(**vars(args))
    lit_module = LitModule.load_from_checkpoint(LAST_CKP, **vars(args))

    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer(resume_from_checkpoint=LAST_CKP)
    print(f'learning rate: {lit_module.learning_rate}')
    trainer.fit(lit_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # trainer related args
    parser = pl.Trainer.add_argparse_args(parser)

    # encoder related args
    parser.add_argument('--enc_dr_rate', type=float, default=0)
    parser.add_argument('--enc_rnn_hidden_dim', type=int, default=6)
    parser.add_argument('--enc_rnn_num_layers', type=int, default=1)
    parser.add_argument('--enc_dim_lat_space', type=int, default=5)
    parser.add_argument('--enc_pretrained', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--enc_fixed_cnn_weights', default=True,
                        action=argparse.BooleanOptionalAction)

    # decoder related args
    parser.add_argument('--dec_num_question_inputs', type=int, default=0)
    parser.add_argument('--dec_hidden_size', type=int, default=10)
    parser.add_argument('--dec_num_hidden_layers', type=int, default=2)
    parser.add_argument('--dec_out_dim', type=int, default=6)

    # filter related args
    parser.add_argument('--filt_initial_log_var', type=float, default=0)
    parser.add_argument('--filt_num_decoders', type=int, default=3)
    parser.add_argument('--formula_exponents', nargs="+",
                        default=[-1, -.5, 0, 1, 2])

    # lit_module related args
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--dl_num_workers', type=int, default=12)
    parser.add_argument('--validdation_split', type=float, default=0.05)
    parser.add_argument('--pretrain_thres', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--lamb', type=float, default=0.1)
    args = parser.parse_args()
    main(args)
