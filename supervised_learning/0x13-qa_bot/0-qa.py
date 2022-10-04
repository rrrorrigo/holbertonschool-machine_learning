#!/usr/bin/env python3
"""Question answering"""


import tensorflow_hub as tfh
import tensorflow as tf
from transformers import BertTokenizer


def question_answer(question, reference):
    """Function that finds a snippet of text within a reference document
    to answer a question"""
    tknzr = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(tknzr)
    model = tfh.load('https://tfhub.dev/see--/bert-uncased-tf2-qa/1')

    question_tokens = tokenizer.tokenize(question)
    paragraph_tokens = tokenizer.tokenize(reference)
    tokens = [len(question)] + question_tokens + [len(reference)] + paragraph_tokens
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_tokens) + 1) + [1] * (len(paragraph_tokens))

    input_word_ids = tf.expand_dims(tf.convert_to_tensor(input_word_ids,
                                                         dtype=tf.int32), 0)
    input_mask = tf.expand_dims(tf.convert_to_tensor(input_mask,
                                                     dtype=tf.int32), 0)
    input_type_ids = tf.expand_dims(tf.convert_to_tensor(input_type_ids,
                                                         dtype=tf.int32), 0)
    outputs = model([input_word_ids, input_mask, input_type_ids])
    # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1
    answer_tokens = tokens[short_start: short_end]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer
