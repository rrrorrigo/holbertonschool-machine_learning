#!/usr/bin/env python3
"""4. Moving Average"""


def moving_average(data, beta):
    """function that calculates the weighted moving average of a data set"""
    i = 0
    avg = []
    avg_act = 0
    while i < len(data):
        avg_act = beta * avg_act + (1 - beta) * data[i]
        avg += [avg_act / (1 - beta ** (i + 1))]
        i += 1
    return avg
