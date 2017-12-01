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
                              createHistogram, makeMultiLineDipoleGraph)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os.path
import plotly
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
    nstates = len(retrieve_hdf5_data(allH5First,'SFS_ENERGIES'))
    natoms = len(retrieve_hdf5_data(allH5First,'CENTER_COORDINATES'))
    print('\nnstates: {} \ndimension: {}'.format(nstates,dime))
    newdime = (dime * 2)-1
    bigArrayLab1 =  np.empty((dime), dtype=object)
    bigArrayLab2 =  np.empty((dime), dtype=object)
    ind = 0
    for fileN in allH5:
        (axis1,str1) = stringTransformation(fileN,0,1)
        (axis2,str2) = stringTransformation(fileN,2,3)
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
               energies = retrieve_hdf5_data(fileN,'SFS_ENERGIES')
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
    nstates = len(retrieve_hdf5_data(allH5First,'SFS_ENERGIES'))
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
        properties = retrieve_hdf5_data(fileN,'PROPERTIES')
        energies = retrieve_hdf5_data(fileN,'SFS_ENERGIES')
        coords = retrieve_hdf5_data(fileN,'CENTER_COORDINATES')
        dmMat = properties[0:3]
        (axis1,str1) = stringTransformation(fileN,0,1)
        (axis2,str2) = stringTransformation(fileN,2,3)
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
    print(labelsAxis1,labelsAxis2)
    #[a,b,c] = [bigArrayB1[0:4], bigArrayB2[0:4], bigArrayE[0:4]]
    #print(a,b,c)
    #[a,b,c] = [bigArrayAxis1, bigArrayAxis2, bigArrayE]
    #[a,b,c] = [bigArrayAxis1, bigArrayAxis2, bigArrayD[:,0,2,:]]
    [a,b,c] = [bigArrayAxis1_2Zero, bigArrayAxis2_2, bigArrayE2Zero]
    np.savetxt('1.txt', a)
    np.savetxt('2.txt', b)
    np.savetxt('3.txt', c)
    # (313,) (313,) (313, 14)
    #splot(a,b,c)
    #plotlyZ(a,b,c)
    #print(a.shape, b.shape, c.shape)
    #mathematicaListGenerator(a,b,c)
    #print(a.shape, b.shape, c.shape)

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

def mathematicaListGenerator(a,b,c):
    '''
    This function takes a,b,c and printout a file that should be copied/pasted
    into Mathematica 11.00
    a :: np.array(ngrid) <- first axis
    b :: np.array(ngrid) <- second axis
    c :: np.array(ngrid,second) <- actual values
    second should be nstates or whatever dimension the data in packed.
    to print vectors like this, every single point needs its coordinates, so we
    have to supply every point like (x,y,z) triplet.
    xclip is the best
    '''
    import string
    import os
    letter = list(string.ascii_lowercase)
    #(length, ) = c.shape
    #surfaces = 1
    (length, surfaces) = c.shape
    finalString = 'ListPlot3D[{'
    matheString = ''
    for sur in np.arange(surfaces):
        fName = letter[sur]
        if ((sur != surfaces-1)):
            finalString = finalString + fName + ','
        else:
            finalString = finalString + fName + '}]'
        stringF = fName + ' = {'
        for ind in np.arange(length):
            first = str(a[ind]) + ','
            second = str(b[ind]) + ','
            third = '{:16.14f}'.format(c[ind,sur])
            if (ind != length-1):
                stringF = stringF+"{"+first+second+third+'},'
            else:
                stringF = stringF+"{"+first+second+third+'}}'
        matheString = matheString + "\n" + stringF
    #print(matheString)
    fn = 'hugeVector'
    with open(fn, "w") as myfile:
        myfile.write(matheString)
    print('\ncat ' + fn + ' | xclip -selection clipboard')
    print('\n'+finalString+'\n')

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

def splot(a,b,c):
    '''
    to generate data and script for gnuplot to generate
    3d plot function.
    a :: np.array(ngrid) <- first axis
    b :: np.array(ngrid) <- second axis
    c :: np.array(ngrid,second) <- actual values
    '''
    (length, surfaces) = c.shape
    fn = 'dataFile'
    fnS = 'GnuplotScript'
    with open(fn, 'w') as outF:
        for i in range(length):
            strC = ' '.join(map(str, c[i]))
            fullString = "{:3.4f} {:3.4f} {} 1\n".format(a[i],b[i],strC)
            if (a[i]-a[i-1] > 0.001):
                outF.write('\n')
            outF.write(fullString)
    with open(fnS, 'w') as scri:
        header = '''#set dgrid3d
#set pm3d
#set style data lines
set xlabel "C1-C5"
set ylabel "C2-C5"
set ticslevel 0
splot '''
        scri.write(header)
        for i in range(surfaces):
            iS = str(i)
            i1 = str(i + 1)
            i3 = str(i + 3)
            cc3 = str(surfaces+3)
            string = '"'+fn+'" u 1:2:'+i3+':($'+cc3+'+'+iS+') t "S_{'+iS+'} '+i1+'"'
            if (i != surfaces-1):
                string = string + ', '
            scri.write(string)


def main():
    ''' Takes a list of rassi files and create graphs on energies and on
    Dipole transition elements '''
    o_inputs = single_inputs("*.rassi.h5", 1)
    inp = read_single_arguments(o_inputs)
    #twoDGraph(inp.glob, inp.proc)
    matrixApproach(inp.glob, inp.proc)


if __name__ == "__main__":
        main()

