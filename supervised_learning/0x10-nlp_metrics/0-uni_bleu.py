#!/usr/bin/env python3
"""unigram BLEU"""


import numpy as np


def P(references, sentence):
    """Function that calculates the probability

    P = common ngram / candidate ngram"""
    references_length = []
    words = {}

    for r in references:
        references_length.append(len(r))
        for word in r:
            if word in sentence and word not in words.keys():
                words[word] = 1

    total = sum(words.values())
    return total


def BP(references, sentence):
    """Function that calculate brevety penalty"""
    sentence_length = len(sentence)
    index = np.argmin([abs(len(i) - sentence_length) for i in references])
    best_match = len(references[index])

    if sentence_length > best_match:
        bp = 1
    else:
        bp = np.exp(1 - float(best_match) / float(sentence_length))
    return bp


def uni_bleu(references, sentence):
    """Function that calculates the unigram BLEU score for a sentence

    references is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score"""
    bp = BP(references, sentence)
    len_s = len(sentence)
    p = P(references, sentence)
    BLEU = bp * np.exp(np.log(p / len_s))
    return BLEU
