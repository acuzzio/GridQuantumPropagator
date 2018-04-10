'''
This module has some quick function to display errors on the command line
'''

import sys

def err(string):
    '''
    print a box with the error in red
    '''
    box('FAIL', string, True)

def good(string):
    '''
    print a box with a green message
    '''
    box('OKGREEN', string, False)

def warning(string):
    '''
    box with yellow message (it does not raise an error)
    '''
    box('WARNING', string, False)

def box(col, string, erro):
    '''
    This is the function that prints boxes on the command line
    '''
    redstring = colors[col] + string + colors['ENDC']
    leng = len(string)
    first = ('*' * (leng+6)) + "\n"
    second = "*" + (' ' * (leng+4)) + '*\n'
    finals = "\n\n" + first + second + "*  " + redstring + "  *\n" + second + first + "\n\n"
    if erro:
        sys.exit(finals)
    else:
        print(finals)

colors = {
    'HEADER' : '\033[95m',
    'OKBLUE' : '\033[94m',
    'OKGREEN' : '\033[92m',
    'WARNING' : '\033[93m',
    'FAIL' : '\033[91m',
    'ENDC' : '\033[0m',
    'BOLD' : '\033[1m',
    'UNDERLINE' : '\033[4m'
}

if __name__ == "__main__":
    print(err('THIS HAPPENED'))


