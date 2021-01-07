from torch import nn
import torchvision
import torch
import src.constants as const


class Encoder(nn.Module):
    def __init__(self,  enc_dr_rate: float = 0.0,
                 enc_rnn_hidden_dim: int = 6, enc_rnn_num_layers: int = 1,
                 num_hidden_states: int = const.NUM_HIDDEN_STATES,
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
        self.fc_out = nn.Linear(enc_rnn_hidden_dim, num_hidden_states)

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


class Decoder(nn.Module):
    def __init__(self, dec_num_question_inputs: int = 1,
                 num_hidden_states: int = const.NUM_HIDDEN_STATES,
                 dec_hidden_size: int = 10,
                 dec_num_hidden_layers: int = 2,
                 dec_single_answer_dim: int = 1,
                 **kwargs):
        super(Decoder, self).__init__()

        self.fc_in = nn.Linear(num_hidden_states + dec_num_question_inputs,
                               dec_hidden_size)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(dec_hidden_size, dec_hidden_size)
             for i in range(dec_num_hidden_layers)])
        self.fc_out = nn.Linear(dec_hidden_size, dec_single_answer_dim)

    def forward(self, lat_space, questions):
        input = torch.cat((lat_space, questions.view(-1, 1)), axis=1)
        output = torch.tanh(self.fc_in(input))
        for h in self.fc_hidden:
            output = torch.tanh(h(output))
        output = self.fc_out(output)
        return output
