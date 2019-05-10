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

<<<<<<< HEAD

=======
>>>>>>> 6482db2f27c75cbc122b668ce0719fb5aa1e3de1
class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...

<<<<<<< HEAD
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.params = nn.ParameterDict({})

        def gate(name, input_dim, num_hidden, p=False):
            # reduce standard deviation: divide by number ofunits
            self.params[name+'Wx'] = nn.Parameter(torch.randn((input_dim, num_hidden)) / num_hidden)
            if p:
                self.params[name+'b'] = nn.Parameter(torch.randn(num_hidden) / num_hidden)
            else:
                self.params[name+'Wh'] = nn.Parameter(torch.randn((num_hidden, num_hidden)) / num_hidden)
                self.params[name+'b'] = nn.Parameter(torch.randn(num_hidden) / num_hidden)

        gate('G', input_dim, num_hidden)
        gate('I', input_dim, num_hidden)
        gate('F', input_dim, num_hidden)
        gate('O', input_dim, num_hidden)
        gate('P', num_hidden, num_classes, p=True)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Implementation here ...
        params = self.params
        c = torch.zeros(1, self.num_hidden).to(self.device)
        h = torch.zeros(1, self.num_hidden).to(self.device)

        x = x.to(self.device)

        for step in range(self.seq_length):
            xx = x[:, step].reshape(-1, self.input_dim)
            g = self.tanh(xx @ params['GWx'] + h @ params['GWh'] + params['Gb'])
            i = self.sig(xx @ params['IWx'] + h @ params['IWh'] + params['Ib'])
            f = self.sig(xx @ params['FWx'] + h @ params['FWh'] + params['Fb'])
            o = self.sig(xx @ params['OWx'] + h @ params['OWh'] + params['Ob'])
            c = g * i + c * f
            h = self.tanh(c) * o

        p = h @ params['PWx'] + params['Pb']

        return p
=======
    def forward(self, x):
        # Implementation here ...
        pass
>>>>>>> 6482db2f27c75cbc122b668ce0719fb5aa1e3de1
