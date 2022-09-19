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


def P(references, grams, n):
    """Function that calculates the probability

    P = common ngram / candidate ngram"""
    reference_grams = []
    words = {}

    for reference in references:
        list_grams = create_ngram(reference, n)
        reference_grams.append(list_grams)

    for r in reference_grams:
        for word in r:
            if word in grams and word not in words.keys():
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


def ngram_bleu(references, sentence, n):
    """Function that calculates the n-gram BLEU score for a sentence:

    references is a list of reference translations
        each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence

    Returns: the unigram BLEU score"""
    gram = create_ngram(sentence, n)
    bp = BP(references, sentence)
    len_s = len(gram)
    p = P(references, gram, n)
    BLEU = bp * np.exp(np.log(p / len_s))
    return BLEU
