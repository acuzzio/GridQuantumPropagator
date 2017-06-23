''' This script reads and make operations on cubes '''

from collections import namedtuple
from argparse import ArgumentParser

import numpy as np
import pandas as pd

def read_this_arguments(single_inputs):
    '''
    input reader for the cube script
    '''
    parser = ArgumentParser()
    parser.add_argument("-c", "--cube",
                        dest="c",
                        type=str,
                        help="cube file (cube)")

    args = parser.parse_args()

    if args.c != None:
        single_inputs = single_inputs._replace(cube=args.g)
    return single_inputs

single_inputs = namedtuple("single_input",
            ("cube",
            )
            )

def main():
    '''
    reads and makes operations on cubes
    '''
    inputs = single_inputs("Density_21.cube", # cube
                           )

    new_inp = read_this_arguments(inputs)
    fn = new_inp.cube
    lol = pd.read_table(fn, delim_whitespace=True,
            skiprows=14).as_matrix().flatten()
    lol = lol[~np.isnan(lol)]  # take out nan values
    print(lol)
    print(lol.shape)

if __name__ == "__main__":
    main()

