#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
from cnn import CNN
from highway import Highway


class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output
            NOTE to self: embed_size IS word_embed_size
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        self.embed_size = embed_size
        self.char_embed_size = 50
        # self.vocab           = vocab
        self.dropout_rate = 0.3
        self.max_word_length = 21

        # dense character embedding
        pad_token_idx = vocab.char2id["<pad>"]
        self.CharEmbedding = nn.Embedding(
            len(vocab.char2id), self.char_embed_size, padding_idx=pad_token_idx
        )

        self.CNN = CNN(self.char_embed_size, self.embed_size, self.max_word_length)
        self.Highway = Highway(self.embed_size)
        self.Dropout = nn.Dropout(self.dropout_rate)

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary
            NOTE to self: this is x_padded in pdf

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        # 1. input, i.e. x_padded -> x_emb (via charEmbedding).
        # (sentence_length, batch_size, max_word_length) -> (sentence_length, batch_size, max_word_length, char_embed_size)
        x_emb = self.CharEmbedding(input)

        # 2. collapse first 2 dimensions and reshape x_emb to x_reshaped
        # (sentence_length, batch_size, max_word_length, char_embed_size) -> (sentence_length * batch_size, char_embed_size, max_word_length)
        sent_len, batch_size, max_word_len, _ = x_emb.shape
        view_shape = (sent_len * batch_size, max_word_len, self.char_embed_size)
        x_reshaped = x_emb.view(view_shape).permute(0, 2, 1)

        # 3. CNN: x_reshaped -> x_conv_out
        # result shape (sentence_length * batch_size, embed_size)
        x_conv_out = self.CNN(x_reshaped)

        # 4. Highway: x_conv_out -> x_highway
        # result shape (sentence_length * batch_size, embed_size). No shape changes
        x_highway = self.Highway(x_conv_out)

        # 5. Dropout: x_highway -> x_word_emb
        x_word_emb = self.Dropout(x_highway)

        # 6. unfold first 2 dimensions
        # (sentence_length * batch_size, embed_size) ->  (sentence_length, batch_size, embed_size)
        x_word_emb = x_word_emb.view(sent_len, batch_size, self.embed_size)

        return x_word_emb
