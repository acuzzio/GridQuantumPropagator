''' This scripts collects transition dipole matrices from several h5 files and
makes graphs '''

from collections import namedtuple
from argparse import ArgumentParser
import glob
import multiprocessing as mp
import numpy as np
from quantumpropagator import (retrieve_hdf5_data, makeJustAnother2Dgraph,
                              createHistogram, makJusAno2DgrMultiline)
import matplotlib.pyplot as plt

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

def graphMultiRassi(globalExp,poolSize):
    ''' collects rassi data and create a elementwise graph '''
    allH5 = sorted(glob.glob(globalExp))
    print(allH5)
    dime = len(allH5)
    nstates = 14
    bigArray = np.empty((dime,3,nstates,nstates))

    ind=0
    for fileN in allH5:
        properties = retrieve_hdf5_data(fileN,'PROPERTIES')
        dmMat = properties[0:3]
        bigArray[ind] = dmMat
        ind += 1

    std = np.std(bigArray[:,0,0,0])
    allstd = np.average(np.abs(bigArray), axis = 0)

    fn = 'heatMap.png'
    transp    = False
    my_dpi    = 150
    ratio     = (9, 16)
    fig, ax1  = plt.subplots(figsize=ratio)

    xticks = np.arange(nstates)+1
    yticks = np.arange(nstates)+1

    plt.subplot(311)
    plt.imshow(allstd[0], cmap='hot', interpolation='nearest')
    plt.subplot(312)
    plt.imshow(allstd[1], cmap='hot', interpolation='nearest')
    plt.subplot(313)
    plt.imshow(allstd[2], cmap='hot', interpolation='nearest')

    #plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi)
    plt.close('all')

    # warning... range(2) excludes Z values
    elems = [[x,y,z] for x in range(2) for y in range(nstates) for z in range(nstates)]

    with mp.Pool(processes=poolSize) as p:
        promises = [ p.apply_async(doThisToEachElement, args=(elem, dime,
            bigArray))
                   for elem in elems ]
        for (p, i) in zip(promises, elems):
            p.get()

    rows = [[x,y] for x in range(2) for y in range(nstates)]

    for row in rows:
        doThisToEachRow(row, dime, bigArray)
    #with mp.Pool(processes=poolSize) as p:
    #    promises = [ p.apply_async(doThisToEachRow, args=(row, dime, bigArray))
    #               for row in rows ]
    #    for (p, i) in zip(promises, rows):
    #        p.get()

def doThisToEachRow(row, dime, bigArray):
    [a,b] = row
    label = str(a+1) + '_' + str(b+1)
    makJusAno2DgrMultiline(np.arange(dime),abs(bigArray[:,a,b]), 'All_from_' +
            label, b)

def doThisToEachElement(elem, dime, bigArray):
    ''' It creates two kind of graphs from the bigarray, elementwise'''
    [a,b,c] = elem
    label = str(a+1) + '_' + str(b+1) + '_' + str(c+1)
    makeJustAnother2Dgraph(np.arange(dime), abs(bigArray[:,a,b,c]), 'Lin_' + label, label)
    createHistogram(np.abs(bigArray[:,a,b,c]), 'His_' + label, binNum=20)


def main():
    ''' Takes a list of rassi files and create graphs on Dipole transition
    elements '''
    inputs = single_inputs("*.rassi.h5", 1)
    new_inp = read_single_arguments(inputs)
    graphMultiRassi(new_inp.glob, new_inp.proc)


if __name__ == "__main__":
        main()


