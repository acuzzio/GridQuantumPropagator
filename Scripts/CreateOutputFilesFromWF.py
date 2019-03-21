''' This is to postprocess a folder full of wavefunctions. '''

import glob
import os

from argparse import ArgumentParser
from quantumpropagator import calculate_stuffs_on_WF, readWholeH5toDict, err

def read_single_arguments():
    '''
    This funcion reads the command line arguments and assign the values on
    the namedTuple for the 3D grid propagator.
    '''
    d = 'This script will launch a Grid quantum propagation'
    parser = ArgumentParser(description=d)
    parser.add_argument("-f", "--folder",
                        dest="f",
                        required=True,
                        type=str,
                        help="This is the folder where the wavefunctions are")

    args = parser.parse_args()

    return args

def check_output_of_Grid(fn):
    '''
    checks if this file first line has 11 number of fields
    if it is 12... this means that this calculation is useless
    '''
    with open(fn,'r') as f:
        first_line = f.readline()
    number_of_line = len(first_line.split(' '))
    if number_of_line != 11:
        err('Watch out. File output already have good format...')


def main():
    '''
    This will launch the postprocessing thing
    '''

    a = read_single_arguments()

    folder = os.path.join(os.path.abspath(a.f), 'Gaussian')
    all_h5 = os.path.join(os.path.abspath(a.f), 'allInput.h5')
    output_of_Grid = os.path.join(os.path.abspath(a.f), 'output')
    output_of_this = os.path.join(os.path.abspath(a.f), 'Output_Abs')

    check_output_of_Grid(output_of_Grid)

    if os.path.isfile(output_of_this):
        print('\n\nrm {}\n\n'.format(output_of_this))
        err('Watch out. File Output_Abs already exists into folder')

    global_expression = folder + '*.h5'

    list_of_wavefunctions = sorted(glob.glob(global_expression))

    all_h5_dict = readWholeH5toDict(all_h5)

    for single_wf in list_of_wavefunctions:
        wf_dict = readWholeH5toDict(single_wf)
        calculate_stuffs_on_WF(wf_dict, all_h5_dict, output_of_this)

if __name__ == "__main__":
    main()
