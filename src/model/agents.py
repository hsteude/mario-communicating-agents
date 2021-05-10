from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, enc_rnn_hidden_dim: int = 6, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(5, 5)
        self.fc = nn.Linear(6*44*44, enc_rnn_hidden_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 6*44*44)
        x = self.fc(x)
        return x


class Encoder(nn.Module):
    def __init__(self,  enc_dr_rate: float = 0.0,
                 enc_rnn_hidden_dim: int = 6, enc_rnn_num_layers: int = 1,
                 enc_dim_lat_space: int = 5,
                 **kwargs):
        super(Encoder, self).__init__()

        self.cnn = SimpleCNN(enc_rnn_hidden_dim=enc_rnn_hidden_dim)
        self.dropout = nn.Dropout(enc_dr_rate)
        self.rnn = nn.LSTM(enc_rnn_hidden_dim, enc_rnn_hidden_dim,
                           enc_rnn_num_layers)
        self.fc_out = nn.Linear(enc_rnn_hidden_dim, enc_dim_lat_space)

    def forward(self, videos):
        b_i, c, ts, h, w = videos.shape
        ts_idx = 0
        y = self.cnn((videos[:, :, ts_idx, :, :]))
        output, (hn, cn) = self.rnn(y.unsqueeze(1))
        for ts_idx in range(1, ts):
            y = self.cnn((videos[:, :, ts_idx, :, :]))
            out, (hn, cn) = self.rnn(y.unsqueeze(1), (hn, cn))
        out = self.dropout(out[:, -1])
        out = self.fc_out(out)
        return out


class Filter(nn.Module):
    def __init__(self, filt_initial_log_var: float = -10,
                 enc_dim_lat_space: int = 5,
                 filt_num_decoders: int = 3,
                 **kwargs):
        super(Filter, self).__init__()

        self.selection_bias = nn.Parameter(torch.tensor(
            np.array([filt_initial_log_var]*(
                enc_dim_lat_space * filt_num_decoders)).reshape(
                filt_num_decoders, enc_dim_lat_space), dtype=torch.float32))

    def forward(self, lat_space, device):
        std = torch.exp(0.5 * self.selection_bias)
        eps = torch.randn(lat_space.shape[0], *std.shape, device=device)
        return [lat_space + std[i, :] * eps[:, i, :]
                for i in range(std.shape[0])]


class Decoder(nn.Module):
    def __init__(self, dec_num_question_inputs: int = 0,
                 enc_dim_lat_space: int = 5,
                 dec_hidden_size: int = 10,
                 dec_num_hidden_layers: int = 2,
                 dec_out_dim: int = 6,
                 **kwargs):
        super(Decoder, self).__init__()

        self.fc_in = nn.Linear(enc_dim_lat_space + dec_num_question_inputs,
                               dec_hidden_size)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(dec_hidden_size, dec_hidden_size)
             for i in range(dec_num_hidden_layers)])
        self.fc_out = nn.Linear(dec_hidden_size, dec_out_dim)

    def forward(self, lat_space, questions):
        # input = torch.cat((lat_space, questions.view(-1, 1)), axis=1)
        questions = questions.view(-1, 1)
        input = torch.cat((lat_space, questions), 1)
        output = torch.tanh(self.fc_in(input))
        for h in self.fc_hidden:
            output = torch.tanh(h(output))
        output = self.fc_out(output)
        return output
