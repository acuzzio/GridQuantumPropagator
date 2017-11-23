'''
This scripts collects energies and transition dipole matrices from several h5 files and
makes graphs.

It is a 1D module
'''

from collections import namedtuple
from argparse import ArgumentParser
import glob
import multiprocessing as mp
import numpy as np
from quantumpropagator import (retrieve_hdf5_data, makeJustAnother2Dgraph,
                              createHistogram, makeMultiLineDipoleGraph,
                              massOf, saveTraj, makeJustAnother2DgraphMULTI,
                              calcAngle)
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
    parser.add_argument("-g", "--graphs",
                        dest="g",
                        action='store_true',
                        help="it creates coords 1d graphs")
    parser.add_argument("-k", "--kinetic",
                        dest="k",
                        action='store_true',
                        help="enables kinetic analysis on 1d")
    args = parser.parse_args()
    if args.n != None:
        single_inputs = single_inputs._replace(glob=args.n)
    if args.p != None:
        single_inputs = single_inputs._replace(proc=args.p)
    if args.g != None:
        single_inputs = single_inputs._replace(graphs=args.g)
    if args.k != None:
        single_inputs = single_inputs._replace(kin=args.k)
    return single_inputs

single_inputs = namedtuple("single_input", ("glob","proc", "kin", "graphs"))

def graphMultiRassi(globalExp,poolSize):
    ''' collects rassi data and create the elementwise graphs '''
    allH5 = sorted(glob.glob(globalExp))
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

    # warning... range(2) excludes Z values
    rows = [[x,y] for x in range(2) for y in range(nstates)]

    for row in rows:
        doThisToEachRow(row, dime, bigArray)
    # For some reason the perallel version of this does not work properly.
    #with mp.Pool(processes=poolSize) as p:
    #    promises = [ p.apply_async(doThisToEachRow, args=(row, dime, bigArray))
    #               for row in rows ]
    #    for (p, i) in zip(promises, rows):
    #        p.get()

def doThisToEachRow(row, dime, bigArray):
    [a,b] = row
    label = str(a+1) + '_' + str(b+1)
    makeMultiLineDipoleGraph(np.arange(dime),abs(bigArray[:,a,b]), 'All_from_' +
            label, b)

def doThisToEachElement(elem, dime, bigArray):
    ''' It creates two kind of graphs from the bigarray, elementwise'''
    [a,b,c] = elem
    label = str(a+1) + '_' + str(b+1) + '_' + str(c+1)
    makeJustAnother2Dgraph(np.arange(dime), abs(bigArray[:,a,b,c]), 'Lin_' + label, label)
    createHistogram(np.abs(bigArray[:,a,b,c]), 'His_' + label, binNum=20)

def kinAnalysis(globalExp, coorGraphs):
    '''
    Takes h5 files from global expression and calculates first derivative and
    second along this coordinate for an attempt at kinetic energy


    For now it only draw graphics...
    '''
    allH5 = sorted(glob.glob(globalExp))
    dime = len(allH5)
    allH5First = allH5[0]
    nstates = len(retrieve_hdf5_data(allH5First,'SFS_ENERGIES'))
    natoms = len(retrieve_hdf5_data(allH5First,'CENTER_COORDINATES'))
    labels = retrieve_hdf5_data(allH5First,'CENTER_LABELS')
    stringLabels = [ b[:1].decode("utf-8") for b in labels ]
    print('\nnstates: {} \ndimension: {}'.format(nstates,dime))

    bigArrayC = np.empty((dime,natoms,3))
    bigArrayE = np.empty((dime,nstates))
    bigArrayA1 = np.empty((dime))

    # fill bigArrayC array
    ind=0
    for fileN in allH5:
        singleCoord = retrieve_hdf5_data(fileN,'CENTER_COORDINATES')
        energies = retrieve_hdf5_data(fileN,'SFS_ENERGIES')
        coords = translateInCM(singleCoord, labels)
        bigArrayC[ind] = coords
        bigArrayE[ind] = energies
        bigArrayA1[ind] = calcAngle(coords,2,3,4)
        ind += 1

    saveTraj(bigArrayC, stringLabels, 'scanGeometriesCMfixed')
    fileNameGraph = 'EnergiesAlongScan'
    makeJustAnother2DgraphMULTI(bigArrayA1, bigArrayE,
            fileNameGraph,'State', 1.0)
    print('\nEnergy graph created:\n\neog ' + fileNameGraph + '.png\n')

    # make graphs
    if coorGraphs:
        for alpha in range(3):
            axes = ['X','Y','Z']
            lab1 = axes[alpha]
            for atomN in range(natoms):
                lab2 = stringLabels[atomN] + str(atomN+1)
                name = 'coord_' + lab1 + '_' + lab2
                makeJustAnother2Dgraph(np.arange(dime), bigArrayC[:,atomN,alpha],name,name)

def translateInCM(geometry, labels):
    '''
    geometry :: (natoms,3) floats <- coordinates in angstrom
    labels :: [Strings] <- atom types
    this function calculates the center of mass and translate the molecule to
    have the origin in it.
    '''

    # Molcas h5 has strings as bytes, so I need to decode the atomLabels to utf-8
    atomMasses = np.array([ massOf(b[:1].decode("utf-8")) for b in labels ])
    rightDimension = np.stack((atomMasses,atomMasses,atomMasses),1)
    geomMass = geometry * rightDimension
    centerOfMass = np.apply_along_axis(np.sum, 0, geomMass)/np.sum(atomMasses)
    translationVector = np.tile(centerOfMass,(15,1))
    newGeom = geometry - translationVector
    return(newGeom)


def main():
    ''' Takes a list of rassi files and create graphs on Dipole transition
    elements '''
    inputs = single_inputs("*.rassi.h5", 1, False, False)
    new_inp = read_single_arguments(inputs)
    if new_inp.kin:
        kinAnalysis(new_inp.glob, new_inp.graphs)
    else:
        graphMultiRassi(new_inp.glob, new_inp.proc)

if __name__ == "__main__":
        main()

