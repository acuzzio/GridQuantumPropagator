'''

This script handles a special case (oxiallyl) and transforms the transition
density matrix into cubes to be visualized.

It is not that hard to generalize, but this workflow is still changing a lot.

For the moment, Gaussian will be used to convert into cube file, using a
template fchk.

'''

import numpy as np

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import (single_point_propagation, printEvenergy,
                              specificPulse, give_me_swapd_oxiallyl,
                              transform_numpy_into_format)
import os


def spawn_grepper(fn):
    '''
    This is a bash script that is needed to extract -> python parsers are slow
      : D

    The new way to do it would be H5.

    '''
    name = "grepper.sh"
    content = '''#!/bin/bash

if [[ -z $1 ]];then
   echo -e "\nYou should use this like:\n\n $ $0 file.out\n"
   exit
fi

fn=$1

for j in 6 5 4 3 2 1
do
   for i in $(seq 1 $j)
   do
       echo "doing TMDZZ for state $j and state $i"
       grep "TDMZZ *$j *$i" $fn | awk '{print $4}' > TDMZZ_${j}_${i}
       wc -l TDMZZ_${j}_${i}
   done
done
'''
    with open(name, 'w') as f:
        f.write(content)
    os.system('/bin/bash ' + name + " " + fn)
    os.system('rm ' + name)


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
                        type=str,
                        help="molcas output file (out)")
    parser.add_argument("-t", "--tdmzz",
                        dest="t",
                        type=str,
                        help="matrix element file TDMZZ")
    parser.add_argument("-o", "--output",
                        dest="o",
                        type=str,
                        help="output file (fchk)")
    parser.add_argument("-e", "--extractMatrix",
                        dest="e",
                        type=str,
                        help="it extracts the TDMZZ matrix from molcas output")

    args = parser.parse_args()

    if args.g != None:
        single_inputs = single_inputs._replace(gaussian_file=args.g)
    if args.m != None:
        single_inputs = single_inputs._replace(molcas_file=args.m)
    if args.t != None:
        single_inputs = single_inputs._replace(TDMZ=args.t)
    if args.o != None:
        single_inputs = single_inputs._replace(out_file=args.o)
    if args.e != None:
        single_inputs = single_inputs._replace(grep=args.e)
    return single_inputs

single_inputs = namedtuple("single_input",
            ("gaussian_file",
             "molcas_file",
             "TDMZ",
             "grep",
             "out_file"
            )
            )

def main():
    '''
    Takes a TDMZZ matrix and cast it into a fchk to create a cube
    '''
    folder = "/home/alessio/Desktop/PERICYCLIC/i-Oxiallyl/"
    inputs = single_inputs(folder + "02-TestGaussian/oxyallylTStemplate.fchk", # gaussian_file
                           folder + "01-SinglePoint/oxyallylTS.out",  # molcas_file
                           folder + "01-SinglePoint/TDMZZ_2_2",       # TDMZ
                           "grepper",                                 # grepper
                           "OUTPUT.fchk"                              # out_file"
                           )

    new_inp = read_this_arguments(inputs)
    if new_inp.grep != "grepper":
        spawn_grepper(new_inp.grep)
   else:
        aon = 128 # problem bound
        initial = np.loadtxt(new_inp.TDMZ).reshape(aon,aon)
        swapped = give_me_swapd_oxiallyl(initial)
        lowtri = swapped[np.tril_indices(aon)]
        rightString = transform_numpy_into_format(lowtri)
        tempfile = "tempfile"
        with open(tempfile, "w") as new_file:
            new_file.write(rightString + "\n")
        bashCommand = "sed '/Total SCF Density/r '" + tempfile + " " + new_inp.gaussian_file + " > " + new_inp.out_file
        #print(bashCommand)
        os.system(bashCommand)
        os.system("rm " + tempfile)

if __name__ == "__main__":
    main()

