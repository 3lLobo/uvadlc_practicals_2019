################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...

<<<<<<< HEAD
        # init Parameters
        self.Whx = nn.Parameter(torch.randn(input_dim, num_hidden), requires_grad=True)
        self.Whh = nn.Parameter(torch.randn(num_hidden, num_hidden), requires_grad=True)
        self.Whp = nn.Parameter(torch.randn(num_hidden, num_classes), requires_grad=True)
        self.bh = nn.Parameter(torch.randn(num_hidden), requires_grad=True)
        self.bp = nn.Parameter(torch.randn(num_classes), requires_grad=True)

        self.time_steps = seq_length
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.device = device

    def forward(self, x):
        # Implementation here ...
        # x input must be a list with ints
        x = x.to(self.device)
        h = torch.zeros((self.num_hidden, self.batch_size)).to(self.device)
        for step in range(self.time_steps):
            xx = torch.matmul(x[:, step].reshape(-1, self.input_dim), self.Whx)
            hh = torch.matmul(h, self.Whh) + self.bh
            h = torch.tanh( xx + hh )

        p = torch.matmul(h, self.Whp) + self.bp
        return p
=======
    def forward(self, x):
        # Implementation here ...
        pass
>>>>>>> 6482db2f27c75cbc122b668ce0719fb5aa1e3de1
