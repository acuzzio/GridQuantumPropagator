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
                              mathematicaListGenerator, gnuSplotCircle)
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

def matrixApproach(globalExp, proc):
    '''

    '''
    allH5 = sorted(glob.glob(globalExp))
    dime = len(allH5)
    allH5First = allH5[0]
    nstates = len(retrieve_hdf5_data(allH5First,'SFS_ENERGIES'))
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
    labelsAxis1 = np.unique(bigArrayLab1)
    labelsAxis2 = np.unique(bigArrayLab2)
    labelsAxis3 = np.unique(bigArrayLab3)
    blenderArray = np.empty((labelsAxis1.size,labelsAxis2.size,
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
                fileNonly = 'zNorbornadiene_' + singleLabel + '.rassi.h5'
                fileN = folder + '/' + fileNonly
                exisT = os.path.exists(fileN)
                print(exisT)
                if exisT:
                    energies = retrieve_hdf5_data(fileN,'SFS_ENERGIES')
                else:
                    energies = np.repeat(-271.0,nstates)
                    print(fileN + ' does not exist...')
                blenderArray[Iax1,Iax2,Iax3] = energies
    axFloat1 = np.array([ labTranform(a) for a in labelsAxis1 ])
    axFloat2 = np.array([ labTranform(b) for b in labelsAxis2 ])
    axFloat3 = np.array([ labTranform(c) for c in labelsAxis3 ])
    axFloat1.tofile('fullA.txt')
    axFloat2.tofile('fullB.txt')
    axFloat3.tofile('fullC.txt')
    blenderArray.tofile('fullD.txt')

def labTranform(string):
    '''
    transform the string of the form
    P014-800
    into his +14.8 float
    '''
    return (float(string.replace('-','.').replace('N','-').replace('P','+')))


def stringTransformation3d(fn):
    '''
    transform the string of the form
    'h5/zNorbornadiene_N006-400_P014-800_P085-500.rassi.h5'
    into 3 numbers and 3 labels
    '''
    fn1 = fn.split('.')[0]  # h5/zNorbornadiene_N006-400_P014-800_P085-500
    # str1 = 'N006-400' ->  axis1 = -6.4
    [str1,str2,str3] = fn1.split('_')[1:]
    [axis1,axis2,axis3] = [
            labTranform(x) for x in
            [str1,str2,str3]]
    return(axis1,str1,axis2,str2,axis3,str3)

single_inputs = namedtuple("single_input", ("glob","proc"))

def main():
    ''' Takes a list of rassi files and create graphs on energies and on
    Dipole transition elements '''
    o_inputs = single_inputs("*.rassi.h5", 1)
    inp = read_single_arguments(o_inputs)
    matrixApproach(inp.glob, inp.proc)

if __name__ == "__main__":
        main()

