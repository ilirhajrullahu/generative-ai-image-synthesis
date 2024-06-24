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


################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################


class VanillaRNN(nn.Module):

    def __init__(
        self, seq_length, input_dim, num_hidden, num_classes, batch_size, device="cpu"
    ):
        super(VanillaRNN, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.W_hx = nn.Parameter(torch.randn(num_hidden, input_dim).to(device))
        self.W_hh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(device))
        self.b_h = nn.Parameter(torch.randn(num_hidden).to(device))
        self.W_hp = nn.Parameter(torch.randn(num_classes, num_hidden).to(device))
        self.b_o = nn.Parameter(torch.randn(num_classes).to(device))

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        
        for t in range(self.seq_length):
            # Convert current input to one-hot encoding
            x_t = x[:, t].unsqueeze(1)
            
            # Update hidden state
            h = torch.tanh(
                torch.einsum('bn,hn->bh', x_t, self.W_hx) +
                torch.einsum('bh,hh->bh', h, self.W_hh) +
                self.b_h
            )
        
        # Compute output using the last hidden state
        output = torch.einsum('bh,ch->bc', h, self.W_hp) + self.b_o
        
        return output