from torch import nn
import torchvision
import torch
import numpy as np
import itertools


class Encoder(nn.Module):
    def __init__(self,  enc_dr_rate: float = 0.0,
                 enc_rnn_hidden_dim: int = 6, enc_rnn_num_layers: int = 1,
                 enc_dim_lat_space: int = 5,
                 enc_pretrained: bool = True,
                 enc_fixed_cnn_weights: bool = True,
                 **kwargs):
        super(Encoder, self).__init__()

        cnn = torchvision.models.resnet18(pretrained=enc_pretrained)
        if enc_fixed_cnn_weights:
            for param in cnn.parameters():
                param.requires_grad = False

        cnn_num_features = cnn.fc.in_features
        cnn.fc = nn.Linear(cnn_num_features, enc_rnn_hidden_dim)

        self.cnn = cnn
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

    def forward(self, lat_space):
        # input = torch.cat((lat_space, questions.view(-1, 1)), axis=1)
        input = lat_space
        output = torch.tanh(self.fc_in(input))
        for h in self.fc_hidden:
            output = torch.tanh(h(output))
        output = self.fc_out(output)
        return output


class FormulaFeatureGenerator(nn.Module):
    """This class takes the latend space and generate polynomial-like feature.
    (actually not a polynomial since also negative exponents are included)

    Number of parameters = permutations of parameter and exponent combinations
    """

    def __init__(self, formula_exponents=['-1', '-.5', '0', '1', '2'],
                 enc_dim_lat_space=4,
                 **kwargs):
        super(FormulaFeatureGenerator, self).__init__()
        self.formula_exponents = [np.float(e) for e in formula_exponents]
        self.small_number = 1e-5
        self.var_list = np.arange(enc_dim_lat_space)
        var_exps_lists = [[(v, e) for e in self.formula_exponents]
                          for v in self.var_list]
        self.combinations = list(itertools.product(*var_exps_lists))
        self.num_features = len(self.combinations)

    def get_feature_names(self):
        assert hasattr(self, 'combinations'), 'Call fit method first!'
        feature_names = \
            ['*'.join([f'x_{c[i][0]}^{c[i][1]}' if c[i][1] != 0 else '1'
                       for i in self.var_list])
             for c in self.combinations]
        return feature_names

    def _create_single_feature(self, X, tup):
        X[X <= self.small_number] = self.small_number
        return torch.tensor([torch.prod(torch.tensor(
            [torch.pow(X[i, j], ei) for j, ei in tup]))
            for i in range(X.shape[0])])

    def forward(self, X):
        X_ = torch.zeros((X.shape[0], len(self.combinations)))
        for i, tup in enumerate(self.combinations):
            X_[:, i] = self._create_single_feature(X, tup)
        return X_


class FormulaDecoder(nn.Module):
    def __init__(self, dec_num_features, dec_out_dim, **kwargs):
        super(FormulaDecoder, self).__init__()
        self.lc = nn.Linear(dec_num_features, dec_out_dim, bias=False)
        bound = 1e-5
        torch.nn.init.uniform(self.lc.weight, a=-bound, b=bound)

    def forward(self, pol_features):
        return self.lc(pol_features)
