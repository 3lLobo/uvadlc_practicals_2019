# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, dropout=0, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        self.lstm_mod = nn.LSTM(
            input_size=vocabulary_size,
            hidden_size=lstm_num_hidden,
            num_layers=lstm_num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.linear_mod = nn.Linear(
            in_features=lstm_num_hidden,
            out_features=vocabulary_size
        )

        self.embed = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=vocabulary_size

        )

    def forward(self, x, hidden=None):
        # Implementation here...
        x = x.long()
        x_onehot = self.embed(x)
        out_lstm, hidden = self.lstm_mod(x_onehot, hidden)

        out_lin = self.linear_mod(out_lstm)
        return out_lin, hidden