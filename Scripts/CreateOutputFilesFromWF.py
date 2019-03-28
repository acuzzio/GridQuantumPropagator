''' This is to postprocess a folder full of wavefunctions. '''

import glob
import numpy as np
import os
import pandas as pd
import pickle

from argparse import ArgumentParser
from quantumpropagator import calculate_stuffs_on_WF, readWholeH5toDict, err
import quantumpropagator as qp

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
    parser.add_argument("-r", "--regions",
                        dest="r",
                        type=str,
                        help="This enables Regions mode and you should indicate regions file")

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

    folder_root = a.f
    folder_Gau = os.path.join(os.path.abspath(a.f), 'Gaussian')
    all_h5 = os.path.join(os.path.abspath(a.f), 'allInput.h5')
    output_of_Grid = os.path.join(os.path.abspath(a.f), 'output')
    output_of_this = os.path.join(os.path.abspath(a.f), 'Output_Abs')
    output_regions = os.path.join(os.path.abspath(a.f), 'Output_Regions.csv')
    output_regionsA = os.path.join(os.path.abspath(a.f), 'Output_Regions')

    if a.r != None:
        regions_file = os.path.abspath(a.r)
        if os.path.isfile(regions_file):
            global_expression = folder_Gau + '*.h5'
            list_of_wavefunctions = sorted(glob.glob(global_expression))

            start_from = 0

            if os.path.isfile(output_regionsA):
                count_output_lines = len(open(output_of_Grid).readlines())
                count_regions_lines = len(open(output_regionsA).readlines())
                count_h5 = len(list_of_wavefunctions)
                if count_regions_lines > count_h5:
                    err('something strange {} -> {}'.format(count_h5,count_regions_lines))
                start_from = count_regions_lines

            with open(regions_file, "rb") as input_file:
                cubess = pickle.load(input_file)

            regionsN = len(cubess)

            print('\n\nI will start from {}\n\n'.format(start_from))
            for fn in list_of_wavefunctions[start_from:]:
                allwf = qp.retrieve_hdf5_data(fn,'WF')
                alltime = qp.retrieve_hdf5_data(fn,'Time')[0]
                outputString_reg = ""
                for r in range(regionsN):
                    uno = allwf[:,:,:,0] # Ground state
                    due = cubess[r]['cube']
                    value = np.linalg.norm(uno*due)**2
                    outputString_reg += " {} ".format(value)
                with open(output_regionsA, "a") as out_reg:
                    print(outputString_reg)
                    out_reg.write(outputString_reg + '\n')


        else:
            err('I do not see the regions file'.format(regions_file))

    else: # regions mode or not?

        check_output_of_Grid(output_of_Grid)

        global_expression = folder_Gau + '*.h5'
        list_of_wavefunctions = sorted(glob.glob(global_expression))

        start_from = 0

        if os.path.isfile(output_of_this):
            count_output_lines = len(open(output_of_Grid).readlines())
            count_abs_lines = len(open(output_of_this).readlines())
            count_h5 = len(list_of_wavefunctions)
            if count_abs_lines > count_h5:
                err('something strange {} -> {} ->  {}'.format(count_h5,count_abs_lines,count_output_lines))
            # if Abs file is there, I need to skip all the wavefunctions I already calculated.
            start_from = count_abs_lines

        all_h5_dict = readWholeH5toDict(all_h5)

        for single_wf in list_of_wavefunctions[start_from:]:
            wf_dict = readWholeH5toDict(single_wf)
            calculate_stuffs_on_WF(wf_dict, all_h5_dict, output_of_this)

if __name__ == "__main__":
    main()
