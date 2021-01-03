from src.model.lit_module import LitModule
import pytorch_lightning as pl
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args):
    # debugging forward pass
    lit_module = LitModule(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
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
    parser.add_argument('--enc_pretrained', default=True,
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--enc_fixed_cnn_weights', default=True,
                        action=argparse.BooleanOptionalAction)

    # lit_module related args
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--dl_num_workers', type=int, default=12)
    parser.add_argument('--validdation_split', type=float, default=0.05)

    args = parser.parse_args()
    main(args)
