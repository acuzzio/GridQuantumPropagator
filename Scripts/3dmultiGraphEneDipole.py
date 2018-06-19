'''
This scripts collects energies and transition dipole matrices from several h5 files and
makes graphs... in 3d... I will NEVER do it
'''

from collections import namedtuple
from argparse import ArgumentParser
import matplotlib
matplotlib.use('TkAgg')
import glob
import multiprocessing as mp
import numpy as np
import os
import re
from quantumpropagator import (retrieve_hdf5_data, makeJustAnother2Dgraph,
                              createHistogram, makeMultiLineDipoleGraph,
                              mathematicaListGenerator, gnuSplotCircle,
                              labTranform, stringTransformation3d)


def read_single_arguments(single_inputs):
    '''
     This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-g", "--globalPattern",
                        dest="g",
                        type=str,
                        required=True,
                        help="it is the global pattern of rassi h5 files")
    parser.add_argument("-p", "--parallel",
                        dest="p",
                        type=int,
                        help="number of processors if you want it parallel")
    args = parser.parse_args()
    if args.g != None:
        single_inputs = single_inputs._replace(glob=args.g)
    if args.p != None:
        single_inputs = single_inputs._replace(proc=args.p)
    return single_inputs

def matrixApproach(globalExp, proc, processed_states):
    '''
    Create blender files from 3d data... meraviglia
    '''
    allH5 = sorted(glob.glob(globalExp))
    dime = len(allH5)
    allH5First = allH5[0]
    nstates = len(retrieve_hdf5_data(allH5First,'ROOT_ENERGIES'))
    natoms = len(retrieve_hdf5_data(allH5First,'CENTER_COORDINATES'))
    print('\nnstates: {} \ndimension: {}'.format(nstates,dime))
    newdime = (dime * 2)-1
    bigArrayLab1 =  np.empty((dime), dtype=object)
    bigArrayLab2 =  np.empty((dime), dtype=object)
    bigArrayLab3 =  np.empty((dime), dtype=object)
    ind = 0
    for fileN in allH5:
        (axis1,str1,axis2,str2,axis3,str3) = stringTransformation3d(fileN)
        bigArrayLab1[ind] = str1
        bigArrayLab2[ind] = str2
        bigArrayLab3[ind] = str3
        ind += 1
    labelsAxis1_b = np.unique(bigArrayLab1)
    # I want ordered phi labels like if they're numbers
    orderedphi = [ x for x in labelsAxis1_b if x[0]=='N' ][::-1] + [ x for x in labelsAxis1_b if x[0]=='P' ]
    labelsAxis1 = np.array(orderedphi, dtype=object)
    labelsAxis2 = np.unique(bigArrayLab2)
    labelsAxis3 = np.unique(bigArrayLab3)
    blenderArray = np.empty((labelsAxis1.size,labelsAxis2.size,
                             labelsAxis3.size,nstates))
    blenderArrayDipo = np.empty((3,processed_states,labelsAxis1.size,labelsAxis2.size,
                             labelsAxis3.size,nstates))
    folder = ('/').join(globalExp.split('/')[:-1])
    if folder == '':
        folder = '.'
    ind = 0
    for Iax1 in range(labelsAxis1.size):
        ax1 = labelsAxis1[Iax1]
        for Iax2 in range(labelsAxis2.size):
            ax2 = labelsAxis2[Iax2]
            for Iax3 in range(labelsAxis3.size):
                ax3 = labelsAxis3[Iax3]
                singleLabel = ax1 + '_' + ax2 + '_' + ax3
                fileNonly = 'zNorbornadiene_' + singleLabel + '.corrected.h5'
                fileN = folder + '/' + fileNonly
                exisT = os.path.exists(fileN)
                #print(exisT)
                if exisT:
                    energies = retrieve_hdf5_data(fileN,'ROOT_ENERGIES')
                    dipole = retrieve_hdf5_data(fileN,'DIPOLES')
                else:
                    energies = np.repeat(-271.0,nstates)
                    print(fileN + ' does not exist...')
                blenderArray[Iax1,Iax2,Iax3] = energies
                for xyz in np.arange(3):
                    for sss in np.arange(processed_states):
                        blenderArrayDipo[xyz,sss,Iax1,Iax2,Iax3] = dipole[xyz,sss,:]
    axFloat1 = np.array([ labTranform(a) for a in labelsAxis1 ])
    axFloat2 = np.array([ labTranform(b) for b in labelsAxis2 ])
    axFloat3 = np.array([ labTranform(c) for c in labelsAxis3 ])
    axFloat1.tofile('fullA.txt')
    axFloat2.tofile('fullB.txt')
    axFloat3.tofile('fullC.txt')
    blenderArray.tofile('fullE.txt')
    for xyz in np.arange(3):
        for sss in np.arange(processed_states):
            elementSlice = blenderArrayDipo[xyz,sss,:]
            labl = {0:'x',1:'y',2:'z'}
            name = 'fullD_' + labl[xyz] + '_' + str(sss) + '.txt'
            elementSlice.tofile(name)

single_inputs = namedtuple("single_input", ("glob","proc"))

def main():
    ''' Takes a list of rassi files and create graphs on energies and on
    Dipole transition elements '''
    o_inputs = single_inputs("*.rassi.h5", 1)
    inp = read_single_arguments(o_inputs)
    matrixApproach(inp.glob, inp.proc, 8)

if __name__ == "__main__":
        main()

