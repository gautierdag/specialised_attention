import torch
import torch.nn as nn
from .positioning_generator import PositioningGenerator


class SpecializedAttention(nn.Module):
    def __init__(self,
                 input_size=50,
                 hidden_size=50,
                 output_size=50,
                 e_dropout=0.0,
                 d_dropout=0.0,
                 pos_gen=True):
        super(Model, self).__init__()

        self.encoder = nn.LSTM(input_size, hidden_size, 1, dropout=e_dropout)

        decoder_input_size = hidden_size + output_size
        self.decoder = nn.LSTM(hidden_size, output_size, 1, dropout=d_dropout)

        self.pos_gen = pos_gen
        if self.pos_gen:
            self.pos_gen = PositioningGenerator(input_size)

    def forward(self, x):
        encoded, hidden = self.encoder(x)

        x = F.relu(self.linear(x))
        return x
