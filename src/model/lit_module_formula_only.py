import torch
import numpy as np
import torchvision
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.dataset import (VideoLabelDataset,
                              VideoFolderPathToTensor,
                              VideoResize)
import src.constants as const
from src.model.agents import (Encoder, FormulaDecoder, Filter,
                              FormulaFeatureGenerator)


class LitModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float = 0.001,
                 ** kwargs):
        super(LitModule, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pretrain = True

        # data set and loader related
        dataset = VideoLabelDataset(
            const.LABELS_TABLE_QA_PATH,
            img_transform=None)
        dataset_size = len(dataset)
        len_val = int(np.floor(dataset_size * self.hparams.validdation_split))
        len_train = dataset_size - len_val
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            dataset=dataset, lengths=[len_train, len_val],
            generator=torch.Generator())

        # filter init
        self.filter = Filter(**self.hparams)

        # polynomial like features
        self.feature_generator = FormulaFeatureGenerator(**self.hparams)

        # decoder init
        # the following is ugly. i did it this way, bcause only attributes of
        # type nn.Module will get send to GPUs (e.g. lists of nn.Modules won't)
        dec_names = [f'dec_{d}' for d in range(self.hparams.filt_num_decoders)]
        for dn in dec_names:
            setattr(self, dn, FormulaDecoder(dec_num_features=self.feature_generator.num_features,
                                             **self.hparams))
        self.decoding_agents = [getattr(self, dn) for dn in dec_names]

    def forward(self, hidden_states):

        lat_space_filt_ls = self.filter(hidden_states, device=self.device)

        out_ls = []
        for dec, ls in zip(self.decoding_agents,  lat_space_filt_ls):
            ls_trans = self.feature_generator(ls)
            out = dec(ls_trans)
            out_ls.append(out)

        out = torch.cat(out_ls, axis=1)
        return out

    def loss_function(self, dec_outs, answers):
        mse_loss = torch.nn.MSELoss()
        answer_loss = mse_loss(dec_outs, answers)
        dec_params = torch.cat([dec.lc.weight for dec in self.decoding_agents])
        param_loss = torch.sum(torch.abs(dec_params))
        return answer_loss + self.hparams.lamb * param_loss

    def training_step(self, batch, batch_idx):

        _, answers, hidden_states, _ = batch
        lat_space_filt_ls = self.filter(hidden_states, device=self.device)

        out_ls = []
        for dec, ls in zip(self.decoding_agents,  lat_space_filt_ls):
            ls_trans = self.feature_generator(ls)
            out = dec(ls_trans)
            out_ls.append(out)

        out = torch.cat(out_ls, axis=1)

        loss = self.loss_function(out, answers)
        self.logger.experiment.add_scalars("losses", {"train_loss": loss})
        return loss

    def validation_step(self, batch, batch_idx):

        _, answers, hidden_states, _ = batch
        lat_space_filt_ls = self.filter(hidden_states, device=self.device)

        out_ls = []
        for dec, ls in zip(self.decoding_agents,  lat_space_filt_ls):
            ls_trans = self.feature_generator(ls)
            out = dec(ls_trans)
            out_ls.append(out)

        out = torch.cat(out_ls, axis=1)

        val_loss = self.loss_function(out, answers)
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

    def log_selection_biases(self):
        """Logs the selection bias for each agent to tensorboard"""
        for i in range(self.hparams.filt_num_decoders):
            self.logger.experiment.add_scalars(
                f'selection_bias_dec{i}',
                {f'lat_neu{j}': self.filter.selection_bias[i, j]
                    for j in range(self.hparams.enc_dim_lat_space)},
                global_step=self.global_step)
