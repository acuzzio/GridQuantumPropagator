''' This script launches a single Point propagation from a h5 file. '''

import numpy as np

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import (single_point_propagation, printEvenergy,
                              specificPulse, give_me_swapd_oxiallyl,
                              transform_numpy_into_format)


def read_this_arguments(single_inputs):
    '''
    This script is for quickly work with cisbutadiene/oxyallyl and print the
    densities
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
    import os
    '''
    Takes a TDMZZ matrix and cast it into a fchk to create a cube
    '''
    folder = "/home/alessio/Desktop/PERICYCLIC/i-Oxiallyl/"
    inputs = single_inputs(folder + "02-TestGaussian/oxyallylTStemplate.fchk", # gaussian_file
                           folder + "01-SinglePoint/oxyallylTS.out",  # molcas_file
                           folder + "01-SinglePoint/TDMZZ_2_2",       # TDMZ
                           "OUTPUT.fchk"                              # out_file"
                           )

    new_inp = read_this_arguments(inputs)

    aon = 128
    initial = np.loadtxt(new_inp.TDMZ).reshape(aon,aon)
    swapped = give_me_swapd_oxiallyl(initial)
    lowtri = swapped[np.tril_indices(aon)]
    rightString = transform_numpy_into_format(lowtri)
    tempfile = "tempfile"
    new_file = open(tempfile, "w")
    new_file.write(rightString + "\n")
    bashCommand = "sed '/Total SCF Density/r '" + tempfile + " " + inputs.gaussian_file + " > " + inputs.out_file
    os.system(bashCommand)
    os.system("rm " + tempfile)

if __name__ == "__main__":
    main()

