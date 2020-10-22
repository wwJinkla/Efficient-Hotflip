#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils


class Highway(nn.Module):
    """Highway network:
    - Highway Networks, Srivastava et al., 2015.
    - https://arxiv.org/abs/1505.00387
    """

    def __init__(self, word_embed_size):
        """
        Initialize Highway model.

        @param word_embed_size (int): Word embedding size (dimensionality) for both the input (x_conv_out) and output
        """
        super(Highway, self).__init__()

        self.word_embed_size = word_embed_size
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
        Take a mini-batch of convolution output, and map them to highway output.

        @param x_conv_out (torch.Tensor): output of convolutional network.
            Shape (batch_size, word_embed_size)
        @returns x_highway (torch.Tensor): combined projection with skip-connection using gate.
            Shape (batch_size, word_embed_size)
        """
        x_proj = torch.relu_(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        return x_highway
