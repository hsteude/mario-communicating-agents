from torch import nn
import torchvision
import src.constants as const


class Encoder(nn.Module):
    def __init__(self,  enc_dr_rate: float = 0.0,
                 enc_rnn_hidden_dim: int = 6, enc_rnn_num_layers: int = 1,
                 enc_num_hidden_states: int = const.NUM_HIDDEN_STATES,
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
        self.fc_out = nn.Linear(enc_rnn_hidden_dim, enc_num_hidden_states)

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
