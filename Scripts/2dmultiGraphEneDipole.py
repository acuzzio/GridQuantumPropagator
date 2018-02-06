'''
This scripts collects energies and transition dipole matrices from several h5 files and
makes graphs.
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os.path
import re

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

def matrixApproach(globalExp, proc):
    '''

    Shitty shitty function, but I need to see if Blender is working

    I try this new way of taking the data. First I watch in the folder, then I
    create a square grid, then I see who is present who is not. This can be
    better to import data in blender and draw surfaces

    globalExp :: String <- with the global expression of the h5 files
    proc :: Int number of processors in case this become heavy and I need to
    parallelize
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
    ind = 0
    for fileN in allH5:
        (axis1,str1) = stringTransformation(fileN,2,3)
        (axis2,str2) = stringTransformation(fileN,4,5)
        bigArrayLab1[ind] = str1
        bigArrayLab2[ind] = str2
        ind += 1
    labelsAxis1 = np.unique(bigArrayLab1)
    labelsAxis2 = np.unique(bigArrayLab2)
    #allLabels = [ ax1 + '_' + ax2 for ax1 in labelsAxis1 for ax2 in labelsAxis2 ]
    blenderArray = np.empty((labelsAxis1.size,labelsAxis2.size,nstates))
    ind = 0
    for Iax1 in range(labelsAxis1.size):
        ax1 = labelsAxis1[Iax1]
        for Iax2 in range(labelsAxis2.size):
            ax2 = labelsAxis2[Iax2]
            singleLabel = ax1 + '_' + ax2
            fileN = re.sub('\*\.', 'Grid_' + singleLabel + '.', globalExp)
            exisT = os.path.exists(fileN)
            if exisT:
               energies = retrieve_hdf5_data(fileN,'ROOT_ENERGIES')
            else:
               energies = np.repeat(-271.0,nstates)
            blenderArray[Iax1,Iax2] = energies
    axFloat1 = np.array([ float(a) for a in labelsAxis1 ])
    axFloat2 = np.array([ float(b) for b in labelsAxis2 ])

    print(axFloat1.shape, axFloat2.shape, blenderArray.shape)
    axFloat1.tofile('fullA.txt')
    axFloat2.tofile('fullB.txt')
    blenderArray.tofile('fullC.txt')

def twoDGraph(globalExp, proc):
    '''
    This function expects a global expression of multiple rassi files
    it will organize them to call a 3d plotting function

    globalExp :: String <- with the global expression of the h5 files
    proc :: Int number of processors in case this become heavy and I need to
    parallelize

    '''
    allH5 = sorted(glob.glob(globalExp))
    dime = len(allH5)
    allH5First = allH5[0]
    nstates = len(retrieve_hdf5_data(allH5First,'ROOT_ENERGIES'))
    natoms = len(retrieve_hdf5_data(allH5First,'CENTER_COORDINATES'))
    print('\nnstates: {} \ndimension: {}'.format(nstates,dime))

    bigArrayD = np.empty((dime,3,nstates,nstates))
    bigArrayE = np.empty((dime,nstates))
    bigArrayC = np.empty((dime,natoms,3))
    bigArrayLab1 =  np.empty((dime), dtype=object)
    bigArrayLab2 =  np.empty((dime), dtype=object)
    bigArrayAxis1 =  np.empty((dime))
    bigArrayAxis2 =  np.empty((dime))
    newdime = (dime * 2)-1
    bigArrayE2 = np.empty((newdime,nstates))
    bigArrayAxis1_2 = np.empty((newdime))
    bigArrayAxis2_2 = np.empty((newdime))
    ind=0
    for fileN in allH5:
        dmMat = retrieve_hdf5_data(fileN,'DIPOLES')
        energies = retrieve_hdf5_data(fileN,'ROOT_ENERGIES')
        coords = retrieve_hdf5_data(fileN,'CENTER_COORDINATES')
        (axis1,str1) = stringTransformation(fileN,2,3)
        (axis2,str2) = stringTransformation(fileN,4,5)
        bigArrayD[ind] = dmMat
        bigArrayE[ind] = energies
        bigArrayE2[ind] = energies
        bigArrayAxis1_2[ind] = axis1
        bigArrayAxis2_2[ind] = axis2
        bigArrayLab1[ind] = str1
        bigArrayLab2[ind] = str2
        if axis2 != 0.0:
           bigArrayE2[ind+(dime-1)] = energies
           bigArrayAxis1_2[ind+(dime-1)] = axis1
           bigArrayAxis2_2[ind+(dime-1)] = -axis2
        else:
           bigArrayE2[ind+(dime-1)] = energies
           bigArrayAxis1_2[ind+(dime-1)] = axis1
           bigArrayAxis2_2[ind+(dime-1)] = axis2
        bigArrayC[ind] = coords
        bigArrayAxis1[ind] = axis1
        bigArrayAxis2[ind] = axis2
        ind += 1
    eneMin = np.min(bigArrayE)
    bigArrayEZero = bigArrayE - eneMin
    bigArrayE2Zero = bigArrayE2 - eneMin
    angleMin = np.min(bigArrayAxis1)
    bigArrayAxis1_2Zero = bigArrayAxis1_2 - angleMin
    labelsAxis1 = np.unique(bigArrayLab1)
    labelsAxis2 = np.unique(bigArrayLab2)
    #print(labelsAxis1,labelsAxis2)
    #[a,b,c] = [bigArrayB1[0:4], bigArrayB2[0:4], bigArrayE[0:4]]
    #print(a,b,c)
    [a,b,c] = [bigArrayAxis1, bigArrayAxis2, bigArrayE]
    print(bigArrayD.shape)
    #[a,b,c] = [bigArrayAxis1, bigArrayAxis2, bigArrayD[:,0,1,:]]
    #[a,b,c] = [bigArrayAxis1_2Zero, bigArrayAxis2_2, bigArrayE2Zero]
    #np.savetxt('1.txt', a)
    #np.savetxt('2.txt', b)
    #np.savetxt('3.txt', c)
    # (313,) (313,) (313, 14)
    #gnuSplotCircle(a,b,c)
    #plotlyZ(a,b,c)
    #print(a.shape, b.shape, c.shape)
    mathematicaListGenerator(a,b,c)
    print(a.shape, b.shape, c.shape)

def flipAndDouble(arr):
    flip = np.concatenate((np.flip(-np.delete(arr,0,0),0),arr),0)
    return(flip)

def stringTransformation(string,firstInd,secondInd):
    '''
    This string Transform is peculiar and problem bound.
    The grid matrix is of the form folder/Grid_120.000_013.966.h5 and
    I need to parse out those 120.000 and 013.966 as Double
    string :: String
    '''
    foundAll = re.findall(r'\d+',string)
    number = float(foundAll[firstInd])+(float(foundAll[secondInd])/1000)
    string = foundAll[firstInd] + '.' + foundAll[secondInd]
    return(number,string)

def plotlyZ(a,b,c):
    '''
    trying to use pltly offline (not easy... also... plotly is not installed by
    the setup), also plotly kind of sucks. Better explore Blender?
    '''
    import plotly.plotly as py
    from plotly.graph_objs import Surface
    (length, surfaces) = c.shape
    plotly.offline.plot([
       #dict(x=[1,2,3],y=[1,2],z=[[1,2,3],[4,5,6]],type='surface'),
       dict(x=a,y=b,z=c[0],type='surface')])
    print(a.shape,b.shape,c.shape)


def main():
    ''' Takes a list of rassi files and create graphs on energies and on
    Dipole transition elements '''
    o_inputs = single_inputs("*.rassi.h5", 1)
    inp = read_single_arguments(o_inputs)
    twoDGraph(inp.glob, inp.proc)
    #matrixApproach(inp.glob, inp.proc)


if __name__ == "__main__":
        main()

