from typing import Dict, List, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model_embeddings import ModelEmbeddings


class CharCNNLSTM(nn.Module):
    """Character-level-CNN LSTM sentence classifier
    """

    def __init__(
        self,
        char_embed_size,
        embed_size,
        hidden_size,
        vocab,
        dropout_rate=0.2,
        num_classes=4,
    ):
        super(CharCNNLSTM, self).__init__()
        self.model_embeddings = ModelEmbeddings(embed_size, char_embed_size, vocab.src)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.h_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.c_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.combined_output_projection = nn.Linear(
            hidden_size * 2 + hidden_size, hidden_size, bias=False
        )
        self.dropout = nn.Dropout(self.dropout_rate)

        self.final_layer = nn.Linear(hidden_size, num_classes, bias=False)
        self.device = "cpu"

    def forward(self, source: List[List[str]]) -> torch.Tensor:
        source_lengths = [len(s) for s in source]
        source_padded_chars = self.vocab.src.to_input_tensor_char(
            source, device=self.device
        )
        enc_hiddens, dec_init_state = self.encode(source_padded_chars, source_lengths)

        logits = self.final_layer(dec_init_state[0])
        return logits

    def encode(
        self, source_padded: torch.Tensor, source_lengths: List[int]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.
        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b, max_word_length), where
                                        b = batch_size, src_len = maximum source sentence length. Note that 
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        X = self.model_embeddings(source_padded)
        X_packed = pack_padded_sequence(X, source_lengths, enforce_sorted=False)
        enc_hiddens, (last_hidden, last_cell) = self.encoder(X_packed)
        (enc_hiddens, _) = pad_packed_sequence(enc_hiddens)
        enc_hiddens = enc_hiddens.permute(1, 0, 2)

        init_decoder_hidden = self.h_projection(
            torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        )
        init_decoder_cell = self.c_projection(
            torch.cat((last_cell[0], last_cell[1]), dim=1)
        )
        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        return enc_hiddens, dec_init_state
