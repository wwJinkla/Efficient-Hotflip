#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.utils


class CNN(nn.Module):
    """Convolutional network"""

    def __init__(self, char_embed_size, num_filters, max_word_length, kernel_size=5):
        """
        Initialize CNN model.

        @param char_embed_size (int): a.k.a the number of input features/channels
        @param num_filters (int): a.k.a the number of output features/channels.
            NOTE: this will be set to word_embed_size in our application
        @param max_word_length (int): maxmimum word length
        @param kernel_size (int): a.k.a window size
        """
        super(CNN, self).__init__()
        self.num_filters = num_filters
        self.char_embed_size = char_embed_size
        self.kernel_size = kernel_size
        self.max_word_length = max_word_length

        self.conv1d = nn.Conv1d(
            in_channels=char_embed_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            bias=True,
        )

        self.maxpool = nn.MaxPool1d(self.max_word_length - kernel_size + 1)

    def forward(self, x_reshaped: torch.Tensor) -> torch.Tensor:
        """
        Compute the convolutional output by
            1. computing x_conv = conv1d(x_reshaped)
            2. applying relu to x_conv
            3. maxpooling across entire the 2nd dimenstion

        @param x_reshaped (torch.Tensor): reshaped tensor of the character embedding tensor (padded).
            Shape (batch_size, char_embed_size, max_word_length)
        @returns x_conv_out (torch.Tensor): convolutional output. Shape (batch_size, word_embed_size)
        """
        # (batch_size, char_embed_size, max_word_length) -> (batch_size, word_embed_size, max_word_length - kernel_size + 1)
        x_conv = self.conv1d(x_reshaped)

        # (batch_size, word_embed_size, max_word_length - kernel_size + 1) - > (batch_size, word_embed_size, 1)
        x_conv_out = self.maxpool(torch.relu_(x_conv))

        # squeeze the 2nd dimension to get (batch_size, word_embed_size)
        return x_conv_out.squeeze(2)
