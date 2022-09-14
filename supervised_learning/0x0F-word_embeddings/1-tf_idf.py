#!/usr/bin/env python3
"""TF-IDF"""


from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """Function that creates a TF-IDF embedding:

    sentences is a list of sentences to analyze
    vocab is a list of the vocabulary words to use for the analysis
        If None, all words within sentences should be used
    Returns: embeddings, features
        embeddings is a numpy.ndarray of shape (s, f) containing the embeddings
            s is the number of sentences in sentences
            f is the number of features analyzed
        features is a list of the features used for embeddings"""
    s_vector = TfidfVectorizer(vocabulary=vocab)
    embeddings = s_vector.fit_transform(sentences)
    embeddings = embeddings.toarray()
    features = s_vector.get_feature_names()
    return embeddings, features
