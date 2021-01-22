import torch
import numpy as np
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.dataset import (VideoLabelDataset,
                              VideoFolderPathToTensor,
                              VideoResize)
import src.constants as const
from src.model.agents import Encoder, Decoder, Filter


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
                 beta: float = 0.001,
                 pretrain_thres: float = 0.001,
                 ** kwargs):
        super(LitModule, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pretrain = True

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
            generator=torch.Generator())

        # encoder init
        self.encoder_agent = Encoder(**self.hparams)

        # decoder init
        # the following is ugly. i did it this way, bcause only attributes of
        # type nn.Module will get send to GPUs (e.g. lists of nn.Modules won't)
        dec_names = [f'dec_{d}' for d in range(self.hparams.filt_num_decoders)]
        for dn in dec_names:
            setattr(self, dn, Decoder(**self.hparams))
        self.decoding_agents = [getattr(self, dn) for dn in dec_names]

        # filter init
        self.filter = Filter(**self.hparams)

    def forward(self, videos):
        out = self.encoder_agent(videos)
        return out

    def loss_function(self, dec_outs, answers, selection_bias, beta):
        mse_loss = torch.nn.MSELoss()
        answer_loss = mse_loss(dec_outs, answers)
        filter_loss = torch.mean(-torch.sum(selection_bias, axis=1))
        return answer_loss + beta * filter_loss

    def training_step(self, batch, batch_idx):
        videos, answers, _, _ = batch
        lat_space = self.encoder_agent(videos)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = torch.cat(dec_outs, axis=1)

        # set beta to 0 and force selection bias to initial value
        # if within pre-training phase (see validation step for phase switch)
        if self.pretrain:
            with torch.no_grad():
                self.filter.selection_bias[:, :] = \
                    torch.empty(*self.filter.selection_bias.shape)\
                    .fill_(self.hparams.filt_initial_log_var)
        beta = 0 if self.pretrain else self.hparams.beta

        loss = self.loss_function(dec_outs, answers,
                                  self.filter.selection_bias, beta)
        self.logger.experiment.add_scalars("losses", {"train_loss": loss})
        self.log_selection_biases()
        return loss

    def validation_step(self, batch, batch_idx):
        videos, answers, _, _ = batch
        lat_space = self.encoder_agent(videos)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = [dec(lat_space) for dec in self.decoding_agents]
        dec_outs = torch.cat(dec_outs, axis=1)
        beta = 0 if self.pretrain else self.hparams.beta
        val_loss = self.loss_function(dec_outs, answers,
                                      self.filter.selection_bias, beta)
        self.logger.experiment.add_scalars("losses", {"val_loss": val_loss})

        # phase switch
        if val_loss < self.hparams.pretrain_thres:
            self.pretrain = False
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

    def log_selection_biases(self):
        """Logs the selection bias for each agent to tensorboard"""
        self.logger.experiment.add_scalars(
            'selection_bias_dec0',
            {'lat_neu0': self.filter.selection_bias[0, 0],
             'lat_neu1': self.filter.selection_bias[0, 1],
             'lat_neu2': self.filter.selection_bias[0, 2],
             'lat_neu3': self.filter.selection_bias[0, 3],
             'lat_neu4': self.filter.selection_bias[0, 4]
             },
            global_step=self.global_step)
        self.logger.experiment.add_scalars(
            'selection_bias_dec1',
            {'lat_neu0': self.filter.selection_bias[1, 0],
             'lat_neu1': self.filter.selection_bias[1, 1],
             'lat_neu2': self.filter.selection_bias[1, 2],
             'lat_neu3': self.filter.selection_bias[1, 3],
             'lat_neu4': self.filter.selection_bias[1, 4]
             },
            global_step=self.global_step)
        self.logger.experiment.add_scalars(
            'selection_bias_dec2',
            {'lat_neu0': self.filter.selection_bias[2, 0],
             'lat_neu1': self.filter.selection_bias[2, 1],
             'lat_neu2': self.filter.selection_bias[2, 2],
             'lat_neu3': self.filter.selection_bias[2, 3],
             'lat_neu4': self.filter.selection_bias[2, 4]
             },
            global_step=self.global_step)
