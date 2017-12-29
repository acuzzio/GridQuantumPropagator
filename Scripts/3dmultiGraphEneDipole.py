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
    print(blenderArray.shape)

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
            float(x.replace('-','.').replace('N','-').replace('P','+')) for x in
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

