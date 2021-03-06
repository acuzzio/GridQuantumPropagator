''' This is to postprocess a folder full of wavefunctions. '''

import glob
import numpy as np
import os
import pandas as pd
import pickle

from argparse import ArgumentParser
from quantumpropagator import calculate_stuffs_on_WF, readWholeH5toDict, err, calculate_dipole_fast_wrapper, warning
import quantumpropagator as qp

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

def difference_mode(file_zero):
    '''
    This function will create a difference folder into the main one
    In this folder, new difference GaussianWF files are created, to visualize the difference
    in population between file_zero (usually after pulse) and the rest.
    '''
    file_zero_abs = os.path.abspath(file_zero)
    wf_file_name_no_ext = os.path.splitext(os.path.basename(file_zero))[0]
    folder_root = os.path.dirname(file_zero_abs)
    difference_folder = os.path.join(folder_root, 'difference_with_{}'.format(wf_file_name_no_ext))
    if not os.path.exists(difference_folder):
        os.mkdir(difference_folder)
    global_expression = folder_root + '/Gaussian*.h5'
    list_of_wavefunctions = sorted(glob.glob(global_expression))

    for wave in list_of_wavefunctions[:]:
        name_this = os.path.splitext(os.path.basename(wave))[0]
        output_file_name = os.path.join(difference_folder, 'diff_{}_{}.h5'.format(wf_file_name_no_ext,name_this))
        difference_this(file_zero_abs,wave,output_file_name)


def difference_this(file_zero_abs, wave, output_file_name):
    '''
    given the three paths, this will create the difference in wavepacket
    '''
    zero = readWholeH5toDict(file_zero_abs)
    other = readWholeH5toDict(wave)
    zero_wf = zero['WF']
    other_wf = other['WF']
    zero_time = zero['Time']
    other_time = other['Time']
    zero_pop = qp.abs2(zero_wf)
    other_pop = qp.abs2(other_wf)
    difference = zero_pop - other_pop
    outputDict = {'WF' : difference, 'Time0' : zero_time, 'Time1' : other_time}
    print('{} cube is done: {} {}'.format(os.path.basename(wave),
                                          np.amax(difference),
                                          np.amin(difference)
                                          ))
    qp.writeH5fileDict(output_file_name, outputDict)

def derivative_mode(folder,every_tot):
    '''
    This function will create a derivative folder into the main one
    In this folder, new difference GaussianWF files are created, to visualize the difference
    in population between file_zero (usually after pulse) and the rest.
    '''
    folder_root = os.path.abspath(folder)
    derivative_folder = os.path.join(folder_root, 'derivative_with_time')
    if not os.path.exists(derivative_folder):
        os.mkdir(derivative_folder)
    global_expression = folder_root + '/Gaussian*.h5'
    list_of_wavefunctions = sorted(glob.glob(global_expression))

    for iwave, (wave1,wave2) in enumerate(zip(list_of_wavefunctions,list_of_wavefunctions[1:])):
        if iwave % every_tot == 0:
            output_file_name = os.path.join(derivative_folder, 'derivative_time_{:04}.h5'.format(iwave))
            difference_this(wave1,wave2,output_file_name)

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
    parser.add_argument("-o", "--output_file",
                        dest="o",
                        type=str,
                        help="This to get new file path")
    parser.add_argument("-d", "--difference",
                        dest="d",
                        type=str,
                        help="This enables Difference mode and you should point wf zero file")
    parser.add_argument("-t", "--time_derivative",
                        dest="t",
                        type=int,
                        help="This enables time derivative mode. It requires an integer to decide how many frames need to be skipped")
    parser.add_argument("-m", "--dipole",
                        dest="m",
                        action='store_true',
                        help="this flag calculates dipoles on the wavefunction")
    args = parser.parse_args()

    return args


def calculate_dipole_slow(wf, all_h5_dict):
    '''
    This function will calculate the dipole in x,y,z components given a wavepacket and an allH5 file
    fn :: FilePath <- the path of the wavefunction
    all_h5_dict :: Dictionary <- the dictionary created by the simulation.
    '''
    pL, gL, tL, nstates = wf.shape

    dipoles = all_h5_dict['dipCube']
    xd = yd = zd = 0.0
    for p in range(15,pL-15):
        for g in range(15,gL-15):
            for t in range(30,tL-30):
                for i in range(nstates):
                    for j in range(i+1):
                        if j != i:
                            # out of diagonal
                            xd += 2 * np.real( np.conj(wf[p,g,t,i]) * wf[p,g,t,j] * dipoles[p,g,t,0,i,j])
                            yd += 2 * np.real( np.conj(wf[p,g,t,i]) * wf[p,g,t,j] * dipoles[p,g,t,1,i,j])
                            zd += 2 * np.real( np.conj(wf[p,g,t,i]) * wf[p,g,t,j] * dipoles[p,g,t,2,i,j])
                        else:
                            # diagonal
                            xd += qp.abs2(wf[p,g,t,i]) * dipoles[p,g,t,0,i,i]
                            yd += qp.abs2(wf[p,g,t,i]) * dipoles[p,g,t,1,i,i]
                            zd += qp.abs2(wf[p,g,t,i]) * dipoles[p,g,t,2,i,i]

    return(xd,yd,zd)


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

    if a.o != None:
        output_dipole = a.o
        center_subcube = (22,22,110)
        extent_subcube = (7,7,20)
        default_tuple_for_cube = (22-7,22+7,22-7,22+7,110-20,110+20)
        # this warning is with respect of NEXT '''if a.o != None:'''
        warning('Watch out, you are using an hardcoded dipole cut {} !!'.format(default_tuple_for_cube))
    else:
        output_dipole = os.path.join(os.path.abspath(a.f), 'Output_Dipole')


    if a.t != None:
        # derivative mode
        print('I will calculate derivatives into {} every {} frames'.format(folder_root, a.t))
        derivative_mode(os.path.abspath(a.f), a.t)

    elif a.d != None:
        # we need to enter Difference mode
        print('I will calculate differences with {}'.format(a.d))
        difference_mode(a.d)

    elif a.m:
        print('We are in dipole mode')
        global_expression = folder_Gau + '*.h5'
        list_of_wavefunctions = sorted(glob.glob(global_expression))

        start_from = 0
        if os.path.isfile(output_dipole): # I make sure that output exist and that I have same amount of lines...
            count_output_lines = len(open(output_of_Grid).readlines())
            count_regions_lines = len(open(output_dipole).readlines())
            count_h5 = len(list_of_wavefunctions)
            if count_regions_lines > count_h5:
                err('something strange {} -> {}'.format(count_h5,count_regions_lines))
            start_from = count_regions_lines

        all_h5_dict = readWholeH5toDict(all_h5)

        print('This analysis will start from {}'.format(start_from))

        for fn in list_of_wavefunctions[start_from:]:
            wf = qp.retrieve_hdf5_data(fn,'WF')
            alltime = qp.retrieve_hdf5_data(fn,'Time')[0]
            #dipx, dipy, dipz = calculate_dipole(wf, all_h5_dict)
            if a.o != None:
                dipx, dipy, dipz, diagx, diagy, diagz, oodiag_x, oodiag_y, oodiag_z = calculate_dipole_fast_wrapper(wf, all_h5_dict, default_tuple_for_cube)
            else:
                dipx, dipy, dipz, diagx, diagy, diagz, oodiag_x, oodiag_y, oodiag_z = calculate_dipole_fast_wrapper(wf, all_h5_dict)
            perm_x = ' '.join(['{}'.format(x) for x in diagx])
            perm_y = ' '.join(['{}'.format(y) for y in diagy])
            perm_z = ' '.join(['{}'.format(z) for z in diagz])
            trans_x = ' '.join(['{}'.format(x) for x in oodiag_x])
            trans_y = ' '.join(['{}'.format(y) for y in oodiag_y])
            trans_z = ' '.join(['{}'.format(z) for z in oodiag_z])
            out_string = '{} {} {} {} {} {} {} {} {} {}'.format(alltime, dipx, dipy, dipz, perm_x, perm_y, perm_z,
                                                                trans_x, trans_y, trans_z)
            # print(output_dipole)
            with open(output_dipole, "a") as out_reg:
                out_reg.write(out_string + '\n')



    elif a.r != None:
        # If we are in REGIONS mode
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
