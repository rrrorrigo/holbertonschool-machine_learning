#!/usr/bin/env python3
"""Answer Questions"""


question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Function that answers questions from a reference text

    reference: is the reference text"""
    exit_response = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        print('Q: ', end='')
        question = input()
        if question.lower() in exit_response:
            print('A: Goodbye')
            break
        response = question_answer(question, reference)
        if response == '' or response is None:
            response = 'Sorry, I do not understand your question.'
        print('A: {}'.format(response))