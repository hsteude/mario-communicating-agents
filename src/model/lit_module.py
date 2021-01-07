import torch
import numpy as np
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.dataset import (VideoLabelDataset,
                              VideoFolderPathToTensor,
                              VideoResize)
import src.constants as const
from src.model.agents import Encoder, Decoder


class LitModule(pl.LightningModule):
    def __init__(self,
                 enc_dr_rate: float = 0,
                 enc_rnn_hidden_dim: int = 6,
                 enc_rnn_num_layers: int = 1,
                 enc_pretrained: bool = True,
                 enc_fixed_cnn_weights: bool = True,
                 learning_rate: float = 0.001,
                 batch_size: float = 12,
                 dl_num_workers: float = 12,
                 validdation_split: float = 0.05,
                 ** kwargs):
        super(LitModule, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # data set and loader related
        dataset = VideoLabelDataset(
            const.LABELS_TABLE_QA_PATH,
            img_transform=torchvision.transforms.Compose([
                VideoFolderPathToTensor(),
                VideoResize(const.IMG_SIZE)]))
        dataset_size = len(dataset)
        len_val = int(np.floor(dataset_size * self.hparams.validdation_split))
        len_train = dataset_size - len_val
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            dataset=dataset, lengths=[len_train, len_val],
            generator=torch.Generator().manual_seed(42))

        # encoder
        self.encoder_agent = Encoder(**self.hparams)

        # decoders
        # self.decoding_agents = [Decoder(**self.hparams)
                                # for _ in range(self.hparams.num_hidden_states)]
        # the following is ugly, but it needs to be done, since 
        dec_names = [f'dec_{d}' for d in range(self.hparams.num_hidden_states)]
        for dn in dec_names:
            setattr(self, dn, Decoder(**self.hparams))
        self.decoding_agents = [getattr(self, dn) for dn in dec_names]

    def forward(self, videos):
        out = self.encoder_agent(videos)
        return out

    def loss_function(self, dec_outs, answers):
        mse_loss = torch.nn.MSELoss()
        answer_loss = mse_loss(dec_outs, answers)
        return answer_loss

    def training_step(self, batch, batch_idx):
        videos, questions, answers, _, _ = batch
        lat_space = self.encoder_agent(videos)
        # TODO inssert filter step here
        dec_outs = [dec(lat_space, questions) for dec in self.decoding_agents]
        dec_outs = torch.cat(dec_outs, axis=1)
        loss = self.loss_function(dec_outs, answers)
        self.logger.experiment.add_scalars("losses", {"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):
        videos, questions, answers, _, _ = batch
        lat_space = self.encoder_agent(videos)
        # TODO inssert filter step here
        dec_outs = [dec(lat_space, questions) for dec in self.decoding_agents]
        dec_outs = torch.cat(dec_outs, axis=1)
        val_loss = self.loss_function(dec_outs, answers)
        self.logger.experiment.add_scalars("losses", {"val_loss": val_loss})
        return val_loss

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.dl_num_workers,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.dl_num_workers,
                          pin_memory=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)
