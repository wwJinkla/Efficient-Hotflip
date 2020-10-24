#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
from collections import Counter
from itertools import chain
from typing import List

import torch

from .utils import pad_sents, pad_sents_char, read_corpus


class VocabEntry(object):
    """ Vocabulary Entry, i.e. structure containing either
    src or tgt language terms.
    """

    def __init__(self, char2id=None):
        """ Init VocabEntry Instance.
        """
        self.char_list = list(
            """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]"""
        )
        if char2id:
            self.char2id = char2id
        else:

            self.char2id = dict()
            self.char2id["<pad>"] = 0
            self.char2id["{"] = 1
            self.char2id["}"] = 2
            self.char2id["<unk>"] = 3
            for i, c in enumerate(self.char_list):
                self.char2id[c] = len(self.char2id)
        self.char_unk = self.char2id["<unk>"]
        self.start_of_word = self.char2id["{"]
        self.end_of_word = self.char2id["}"]
        assert self.start_of_word + 1 == self.end_of_word

        self.id2char = {v: k for k, v in self.char2id.items()}

    def __getitem__(self, character):
        """ Retrieve character's index. Return the index for the unk
        token if the character is out of vocabulary.
        @param character (str): character to look up.
        @returns index (int): index of character 
        """
        return self.char2id.get(character, self.char_unk)

    def __contains__(self, character):
        """ Check if character is captured by VocabEntry.
        @param character (str): character to look up
        @returns contains (bool): whether character is contained    
        """
        return char in self.char2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the VocabEntry.
        """
        raise ValueError("vocabulary is readonly")

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.char2id)

    def __repr__(self):
        """ Representation of VocabEntry to be used
        when printing the object.
        """
        return "Vocabulary[size=%d]" % len(self)

    def words2charindices(self, sents):
        """ Convert list of sentences of words into list of list of list of character indices.
        @param sents (list[list[str]]): sentence(s) in words
        @return word_ids (list[list[list[int]]]): sentence(s) in indices
        """
        ###     This method should convert characters in the input sentences into their
        ###     corresponding character indices using the character vocabulary char2id
        ###     defined above.
        ###
        ###     You must prepend each word with the `start_of_word` character and append
        ###     with the `end_of_word` character.
        word_ids = []
        for sent in sents:
            word_list = []
            for word in sent:  # ["Hello", "World"]

                char_list = [self.start_of_word]
                for char in word:  # ["H", "e", "l", "l", "o"]
                    char_list.append(self.char2id[char])
                char_list.append(self.end_of_word)

                word_list.append(char_list)
            word_ids.append(word_list)
        return word_ids

    def to_input_tensor_char(
        self, sents: List[List[str]], max_word_length: int, device: torch.device
    ) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.

        @param sents (List[List[str]]): list of sentences (words)
        @param device: device on which to load the tensor, i.e. CPU or GPU

        @returns sents_var: tensor of (max_sentence_length, batch_size, max_word_length)
        """
        ###     Connect `words2charindices()` and `pad_sents_char()` which you've defined in
        ###     previous parts
        sents_id = self.words2charindices(sents)
        sents_id = pad_sents_char(
            sents_id, self.char2id["<pad>"], max_word_length=max_word_length
        )
        try:
            sents_id = torch.tensor(sents_id, dtype=torch.long, device=device).permute(
                1, 0, 2
            )
        except ValueError:
            for sent in sents_id:
                for word in sent:
                    print(len(word), word)
        return sents_id


class Vocab(object):
    """ Vocab encapsulating src and target langauges.
    """

    def __init__(self, src_vocab: VocabEntry):
        """ Init Vocab.
        @param src_vocab (VocabEntry): VocabEntry for source language
        """
        self.src = src_vocab

    @staticmethod
    def build(char2id=None) -> "Vocab":
        src = VocabEntry(char2id)
        return Vocab(src)

    def save(self, file_path):
        """ Save Vocab to file as JSON dump.
        @param file_path (str): file path to vocab file
        """
        json.dump(
            dict(char2id=self.src.char2id,), open(file_path, "w"), indent=2,
        )

    @staticmethod
    def load(file_path):
        """ Load vocabulary from JSON dump.
        @param file_path (str): file path to vocab file
        @returns Vocab object loaded from JSON dump
        """
        entry = json.load(open(file_path, "r"))
        char2id = entry.get("char2id")
        return Vocab(VocabEntry(char2id))

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return "Vocab(source %d character)" % (len(self.src),)
