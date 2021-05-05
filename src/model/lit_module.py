import torch
import pytorch_lightning as pl
from src.model.agents import Encoder, Decoder, Filter


class LitModule(pl.LightningModule):
    def __init__(self, *args, ** kwargs):
        super(LitModule, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = self.hparams.learning_rate
        self.pretrain = True

        # encoder init
        self.encoder_agent = Encoder(**self.hparams)

        # decoder init
        # the following is ugly. I did it this way, bcause only attributes of
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
        filter_loss = -torch.sum(selection_bias)
        return answer_loss + beta * filter_loss

    def _shared_eval(self, videos):
        lat_space = self.encoder_agent(videos)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = torch.cat(dec_outs, axis=1)
        return dec_outs

    def training_step(self, batch, batch_idx):
        videos, answers, hidden_states, _ = batch
        dec_outs = self._shared_eval(videos)

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
        videos, answers, hidden_states, _ = batch
        dec_outs = self._shared_eval(videos)
        beta = 0 if self.pretrain else self.hparams.beta
        val_loss = self.loss_function(dec_outs, answers,
                                      self.filter.selection_bias, beta)
        val_loss = self.loss_function(dec_outs, answers,
                                      self.filter.selection_bias, beta)
        self.logger.experiment.add_scalars("losses", {"val_loss": val_loss})

        # phase switch
        if val_loss < self.hparams.pretrain_thres:
            self.pretrain = False
        return val_loss

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
