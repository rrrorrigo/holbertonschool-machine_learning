#!/usr/bin/env python3

def cat_arrays(arr1, arr2):
    """function that concatenates two arrays"""
    concatArr = list(arr1)
    for i in range(len(arr2)):
        concatArr.append(arr2[i])
    return concatArr