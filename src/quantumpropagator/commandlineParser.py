'''
This module takes care of argument parsing.
'''

from collections import namedtuple
from argparse import ArgumentParser

def readArguments(inputs):
    '''
    This funcion reads the command line arguments and assign the values on
    the namedTuple.
    '''
    parser = ArgumentParser()

    parser.add_argument("-n", "--name",
                        dest="n",
                        type=str,
                        help="Name of the project")
    parser.add_argument("-s", "--totalSteps",
                        dest="s",
                        type=int,
                        help="Total time of propagation")
    parser.add_argument("-d", "--deltaTime",
                        dest="d",
                        type=float,
                        help="Time step of propagation")
    args = parser.parse_args()

    if args.n != None:
        inputs = inputs._replace(label=args.n)
    if args.s != None:
        inputs = inputs._replace(fullTime=args.s)
    if args.d != None:
        inputs = inputs._replace(timeStep=args.d)

    return inputs

if __name__ == "__main__":
    inputData = namedtuple("inputData",("timeStep","fullTime","label"))
    inputs = inputData(0.04,10,"coi")
    a = readArguments(inputs)
    print(a)
