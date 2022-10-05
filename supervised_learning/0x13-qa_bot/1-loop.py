#!/usr/bin/env python3
"""Create the loop"""


if __name__ == '__main__':
    exit_response = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        print('Q: ', end='')
        answer = input()
        if answer.lower() in exit_response:
            print('A: Goodbye')
            break
        print('A: ')
