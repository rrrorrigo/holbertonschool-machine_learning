#!/usr/bin/env python3
"""Gensim to keras"""


def gensim_to_keras(model):
    """Function that converts a gensim word2vec model to a keras
    Embedding layer:

    model is a trained gensim word2vec models

    Returns: the trainable keras Embedding"""
    return model.wv.get_keras_embedding(train_embeddings=False)
