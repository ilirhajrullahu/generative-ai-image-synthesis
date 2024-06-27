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


class LSTM(nn.Module):

    def __init__(
        self, seq_length, input_dim, num_hidden, num_classes, batch_size, device="cpu"
    ):
        super(LSTM, self).__init__()
        # Initialization here ...
                
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.Wgx = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Wgh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.bg = nn.Parameter(torch.randn(num_hidden))

        self.Wix = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Wih = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.bi = nn.Parameter(torch.randn(num_hidden))

        self.Wfx = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Wfh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.bf = nn.Parameter(torch.randn(num_hidden))

        self.Wox = nn.Parameter(torch.randn(num_hidden, input_dim))
        self.Woh = nn.Parameter(torch.randn(num_hidden, num_hidden))
        self.bo = nn.Parameter(torch.randn(num_hidden))

        # Initialize weights and biases for output mapping
        self.Wph = nn.Parameter(torch.randn(num_classes, num_hidden))
        self.bp = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        h_t = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        c_t = torch.zeros(self.batch_size, self.num_hidden).to(self.device)
        
        for t in range(self.seq_length):
            x_t = x[:, t].unsqueeze(1)
            
            g_t = torch.tanh(x_t @ self.Wgx.t() + h_t @ self.Wgh.t() + self.bg)
            i_t = torch.sigmoid(x_t @ self.Wix.t() + h_t @ self.Wih.t() + self.bi)
            f_t = torch.sigmoid(x_t @ self.Wfx.t() + h_t @ self.Wfh.t() + self.bf)
            o_t = torch.sigmoid(x_t @ self.Wox.t() + h_t @ self.Woh.t() + self.bo)
            
            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        p_t = h_t @ self.Wph.t() + self.bp
        return p_t
