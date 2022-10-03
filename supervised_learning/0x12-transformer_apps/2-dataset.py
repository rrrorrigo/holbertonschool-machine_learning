#!/usr/bin/env python3
"""Dataset"""


import tensorflow_datasets as tfds
import tensorflow as tf


class Dataset:
    """Dataset class that loads and preps a dataset for machine translation"""

    def __init__(self):
        """Class constructor

        creates the instance attributes:
        data_train, which contains the ted_hrlr_translate/pt_to_en
         tf.data.Dataset train split, loaded as_supervided
        data_valid, which contains the ted_hrlr_translate/pt_to_en
         tf.data.Dataset validate split, loaded as_supervided
        tokenizer_pt is the Portuguese tokenizer created from the training set
        tokenizer_en is the English tokenizer created from the training set"""
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)
        data = self.data_train
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(data)
        pt_tok, en_tok = self.tokenizer_pt, self.tokenizer_en
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Function that creates sub-word tokenizers for the dataset

        data is a tf.data.Dataset whose examples are formatted
        as a tuple (pt, en)
        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence

        The maximum vocab size should be set to 2**15
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt is the Portuguese tokenizer
            tokenizer_en is the English tokenizer"""
        tokenizer_pt = tfds.deprecated.text\
                           .SubwordTextEncoder\
                           .build_from_corpus(
                            (pt.numpy() for pt, en in data), 2**15)
        tokenizer_en = tfds.deprecated.text\
                           .SubwordTextEncoder\
                           .build_from_corpus(
                            (en.numpy() for pt, en in data), 2**15)
        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Function that encodes a translation into tokens

        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence

        Returns: pt_tokens, en_tokens
            pt_tokens is a np.ndarray containing the Portuguese tokens
            en_tokens is a np.ndarray. containing the English tokens"""
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size
        pt_tokens = [pt_vocab_size] + self.tokenizer_pt.encode(pt.numpy()) + \
                    [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + self.tokenizer_en.encode(en.numpy()) + \
                    [en_vocab_size + 1]
        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Function that acts as a tensorflow wrapper for the encode
        instance method

        pt is the tf.Tensor containing the Portuguese sentence
        en is the tf.Tensor containing the corresponding English sentence"""
        pt_tf, en_tf = tf.py_function(self.encode, (pt, en), (tf.int64,
                                                              tf.int64))

        return pt_tf, en_tf
