
import sys

def err(string):
    box('FAIL', string, True)

def good(string):
    box('OKGREEN', string, False)

def box(col,string,erro):
    redstring = colors[col] + string + colors['ENDC']
    leng      = len(string)
    first     = ('*' * (leng+6)) + "\n"
    second    = "*" + (' ' * (leng+4)) + '*\n'
    finalS    = "\n\n" + first + second + "*  " + redstring + "  *\n" + second + first + "\n\n"
    if erro:
       sys.exit(finalS)
    else:
       print(finalS)

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


