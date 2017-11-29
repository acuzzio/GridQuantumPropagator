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

def twoDGraph(globalExp, proc):
    '''
    This function expects a global expression of multiple rassi files
    it will organize them to call a 3d plotting function
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
    bigArrayLab1 =  np.empty((dime))
    bigArrayLab2 =  np.empty((dime))

    ind=0
    for fileN in allH5:
        properties = retrieve_hdf5_data(fileN,'PROPERTIES')
        energies = retrieve_hdf5_data(fileN,'SFS_ENERGIES')
        coords = retrieve_hdf5_data(fileN,'CENTER_COORDINATES')
        dmMat = properties[0:3]
        lab1 = stringTransformation(fileN,0,1)
        lab2 = stringTransformation(fileN,2,3)
        bigArrayD[ind] = dmMat
        bigArrayE[ind] = energies
        bigArrayC[ind] = coords
        bigArrayLab1[ind] = lab1
        bigArrayLab2[ind] = lab2
        ind += 1
#    #[a,b,c] = [bigArrayB1[0:4], bigArrayB2[0:4], bigArrayE[0:4]]
#    #print(a,b,c)
    [a,b,c] = [bigArrayLab1, bigArrayLab2, bigArrayE]
#    # (313,) (313,) (313, 14)
#    #print(a.shape, b.shape, c.shape)
#    #splot(a,b,c)
    #plotlyZ(a,b,c)
    mathematicaListGenerator(a,b,c)

def stringTransformation(string,firstInd,secondInd):
    '''
    This string Transform is peculiar and problem bound.
    The grid matrix is of the form folder/Grid_120.000_013.966.h5 and
    I need to parse out those 120.000 and 013.966 as Double
    string :: String
    '''
    foundAll = re.findall(r'\d+',string)
    number = float(foundAll[firstInd])+(float(foundAll[secondInd])/1000)
    return(number)

def mathematicaListGenerator(a,b,c):
    import string
    letter = list(string.ascii_lowercase)
    (length, surfaces) = c.shape
    finalString = 'ListPlot3D[{'
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
            third = str(c[ind,sur])
            if (ind != length-1):
                stringF = stringF+"{"+first+second+third+'},'
            else:
                stringF = stringF+"{"+first+second+third+'}}'
        print(stringF)
    print('\n'+finalString+'\n')

def plotlyZ(a,b,c):
    '''
    trying to use pltly offline (not easy... also... plotly is not installed by
    the setup)
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
    twoDGraph(inp.glob, inp.proc)


if __name__ == "__main__":
        main()

