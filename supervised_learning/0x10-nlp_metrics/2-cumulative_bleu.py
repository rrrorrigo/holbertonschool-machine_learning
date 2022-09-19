#!/usr/bin/env python3
"""unigram BLEU"""


import numpy as np


def create_ngram(sentence, n):
    """Function that create ngram from sentence"""
    list_grams_cand = []
    for i in range(len(sentence)):
        last = i + n
        begin = i
        if last >= len(sentence) + 1:
            break
        aux = sentence[begin: last]
        result = ' '.join(aux)
        list_grams_cand.append(result)
    return list_grams_cand


def P(references, sentence, n):
    """Function that calculates the probability

    P = common ngram / candidate ngram"""
    reference_grams = []
    grams = set(create_ngram(sentence, n))
    len_g = len(grams)
    words = {}

    for reference in references:
        list_grams = create_ngram(reference, n)
        reference_grams.append(list_grams)

    for r in reference_grams:
        for word in r:
            if word in grams and word not in words.keys():
                words[word] = 1

    total = sum(words.values())
    return total / len_g


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


def cumulative_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence:

    references is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score"""
    prob = []
    for i in range(1, n + 1):
        result = P(references, sentence, i)
        prob.append(result)
    bp = BP(references, sentence)
    BLEU = bp * np.exp(np.sum(np.log(prob) / n))
    return BLEU
