''' This script launches a single Point propagation from a h5 file. '''

import numpy as np

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import (single_point_propagation, printEvenergy,
                              specificPulse, give_me_swapd_oxiallyl,
                              chunksOfList)


def read_this_arguments(single_inputs):
    '''
    This script is for quickly work with cisbutadiene/oxyallyl
    '''
    parser = ArgumentParser()
    parser.add_argument("-g", "--gaussian",
                        dest="g",
                        type=str,
                        help="gaussian template file (fchk)")
    parser.add_argument("-m", "--molcas",
                        dest="m",
                        type=int,
                        help="molcas output file (out)")
    parser.add_argument("-o", "--output",
                        dest="o",
                        type=float,
                        help="output file (fchk)")

    args = parser.parse_args()

    if args.g != None:
        single_inputs = single_inputs._replace(out_folder=args.g)
    if args.m != None:
        single_inputs = single_inputs._replace(nsteps=args.m)
    if args.o != None:
        single_inputs = single_inputs._replace(dt=args.o)
    return single_inputs

single_inputs = namedtuple("single_input",
            ("gaussian_file",
             "molcas_file",
             "TDMZ",
             "out_file"
            )
            )

def main():
    '''
    Launches a single Point propagation from a h5 file.
    '''
    inputs = single_inputs("/home/alessio/Desktop/PERICYCLIC/i-Oxiallyl/02-TestGaussian/oxyallylTS.fchk",         # gaussian_file
                           "/home/alessio/Desktop/PERICYCLIC/i-Oxiallyl/01-SinglePoint/oxyallylTS.out",           # molcas_file
                           "/home/alessio/Desktop/PERICYCLIC/i-Oxiallyl/01-SinglePoint/TDMZZ_1_1",
                           # TDMZ
                           "OUTPUT.fchk"                                                                          # out_file"
                           )

    new_inp = read_this_arguments(inputs)

    fGauDou = '{:15.8E}'
    aon = 128
    a = np.loadtxt(new_inp.TDMZ).reshape(aon,aon)
    b = give_me_swapd_oxiallyl(a)
    c = np.tril_indices(aon)
    d = b[c]
    unlines = lambda ys: '\n'.join(ys)
    unwords = lambda ys: ' '.join(ys)
    dd = ['{:15.8E}'.format(i) for i in d]
    ddd =list(chunksOfList(dd,5))
    dddd = unlines([unwords(x) for x in ddd])
    print(dddd)


if __name__ == "__main__":
    main()

