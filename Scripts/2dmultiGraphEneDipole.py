'''
This scripts collects energies and transition dipole matrices from several h5 files and
makes graphs.
'''

from collections import namedtuple
from argparse import ArgumentParser
import glob
import multiprocessing as mp
import numpy as np
from quantumpropagator import (retrieve_hdf5_data, makeJustAnother2Dgraph,
                              createHistogram, makeMultiLineDipoleGraph,
                              calcBond, fromBohToAng)
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d

def read_single_arguments(single_inputs):
    '''
     This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-n", "--globalPattern",
                        dest="n",
                        type=str,
                        required=True,
                        help="it is the global pattern of rassi h5 files")
    parser.add_argument("-p", "--parallel",
                        dest="p",
                        type=int,
                        help="number of processors if you want it parallel")
    args = parser.parse_args()
    if args.n != None:
        single_inputs = single_inputs._replace(glob=args.n)
    if args.p != None:
        single_inputs = single_inputs._replace(proc=args.p)
    return single_inputs

single_inputs = namedtuple("single_input", ("glob","proc"))

def twoDGraph(globalExp, proc):
    allH5 = sorted(glob.glob(globalExp))
    dime = len(allH5)
    allH5First = allH5[0]
    nstates = len(retrieve_hdf5_data(allH5First,'SFS_ENERGIES'))
    natoms = len(retrieve_hdf5_data(allH5First,'CENTER_COORDINATES'))
    print('\nnstates: {} \ndimension: {}'.format(nstates,dime))

    bigArrayD = np.empty((dime,3,nstates,nstates))
    bigArrayE = np.empty((dime,nstates))
    bigArrayC = np.empty((dime,natoms,3))
    bigArrayB1 = np.empty((dime))
    bigArrayB2 = np.empty((dime))

    ind=0
    for fileN in allH5:
        properties = retrieve_hdf5_data(fileN,'PROPERTIES')
        energies = retrieve_hdf5_data(fileN,'SFS_ENERGIES')
        coords = retrieve_hdf5_data(fileN,'CENTER_COORDINATES')
        dmMat = properties[0:3]
        bond1 = fromBohToAng(calcBond(coords,1,5))
        bond2 = fromBohToAng(calcBond(coords,2,5))
        bigArrayD[ind] = dmMat
        bigArrayE[ind] = energies
        bigArrayC[ind] = coords
        bigArrayB1[ind] = bond1
        bigArrayB2[ind] = bond2
        ind += 1

    [a,b,c] = [bigArrayB1[0:3], bigArrayB2[0:3], bigArrayE[0:3]]
    splot(a,b,c)

def splot(a,b,c):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = axes3d.get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    print(matplotlib.rcParams['backend'])
    plt.show()

def main():
    ''' Takes a list of rassi files and create graphs on energies and on
    Dipole transition elements '''
    o_inputs = single_inputs("*.rassi.h5", 1)
    inp = read_single_arguments(o_inputs)
    twoDGraph(inp.glob, inp.proc)


if __name__ == "__main__":
        main()

