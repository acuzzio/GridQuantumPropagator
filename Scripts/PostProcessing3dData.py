'''
This scripts takes care of postprocessing and reorganizing/correcting
data after Molcas scan...
'''

from collections import namedtuple
from argparse import ArgumentParser

import glob
from itertools import repeat
import multiprocessing
import numpy as np
import os

from quantumpropagator import (retrieve_hdf5_data, writeH5file,
                       npArrayOfFiles, printMatrix2D, createTabellineFromArray,
                       writeH5fileDict, readWholeH5toDict, chunksOf)


def read_single_arguments(single_inputs):
    '''
     This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-o", "--outputFolder",
                        dest="o",
                        type=str,
                        help="The location of the output folder")
    parser.add_argument("-s", "--scanFolder",
                        dest="s",
                        type=str,
                        help="The location of the Grid folder")
    parser.add_argument("-p", "--parallel",
                        dest="p",
                        type=int,
                        help="number of processors if you want it parallel")
    parser.add_argument("-d", "--displacement",
                        dest="d",
                        nargs='+',
                        help="Corrector, takes two arguments:\n"
                        "Expression of h5 files\n"
                        "Folder of outputs\n")


    args = parser.parse_args()
    if args.o != None:
        single_inputs = single_inputs._replace(outF=args.o)
    if args.s != None:
        single_inputs = single_inputs._replace(glob=args.s)
    if args.p != None:
        single_inputs = single_inputs._replace(proc=args.p)
    if args.d != None:
        single_inputs = single_inputs._replace(direction=args.d)

    return single_inputs


def createOutputFile(tupleI):
    '''
    given a projectname, creates single outFile (not yet corrected)
    tupleI :: (String,String) <- is composed by (fold,outFol)
    fold :: String <- the folder of a single calculation
    outFol :: String <- the folder where the output is collected
    '''
    (fold,outFol) = tupleI
    print('doing ' + fold)
    folder = os.path.dirname(fold)
    proj = os.path.basename(fold)
    root = folder +'/' + proj + '/' + proj
    oroot = os.path.join(outFol,proj)
    h5rasscf = root + '.rasscf.h5'
    h5dipole = root + '.transDip.h5'
    h5rassi = root + '.rassi.h5'
    h5out = root + '.out'
    a_exist = all([ os.path.isfile(f) for f in
        [h5rassi,h5rasscf,h5out,h5dipole]])
    if a_exist:
        log = proj + ': all files present'

        boolean = True

        [geom,aType,ciVect,ener] = retrieve_hdf5_data(h5rasscf,
                                   ['CENTER_COORDINATES',
                                    'CENTER_LABELS',
                                    'CI_VECTORS',
                                    'ROOT_ENERGIES'])

        [dipoles, transDen] = retrieve_hdf5_data(h5dipole,
                              ['SFS_EDIPMOM',
                               'SFS_TRANSITION_DENSITIES'])
        [overlap] = retrieve_hdf5_data(h5rassi,
                           ['ORIGINAL_OVERLAPS'])
        nstates = ener.size
        natoms = aType.size
        # I want to save only the low left corner of overlap
        NAC = parseNAC(h5out,nstates,natoms)
        outfile = oroot + '.all.h5'
        outTuple = [('CENTER_COORDINATES', geom),
                    ('CENTER_LABELS', aType),
                    ('CI_VECTORS', ciVect),
                    ('ROOT_ENERGIES', ener),
                    ('DIPOLES', dipoles),
                    #('TRANDENS',transDen),
                    ('OVERLAP', overlap[nstates:,:nstates]),
                    ('NAC', NAC)]

        try:
           writeH5file(outfile,outTuple)
        except "Unable to open file":
           print(outfile)
        # count_nonzero does not work anymore with the new thing
        log += ' -> ' + str(np.count_nonzero(NAC)/2)
    else:
        log = proj + ': this calculation is not completed'
        boolean = False
    return(log,boolean)


def parseNAC(fileN,nstates,natoms):
    '''
    I use bash here to quickly get out NAC values
    fileOut :: filePath <- the output of molcas
    '''
    import subprocess
    emptyMat = np.zeros((nstates,nstates,natoms,3))
    # This is the most problem Bound function I can think of (bash) 
    command = 'grep -A22 "Total derivative coupling" ' + fileN + " | grep -B14 \"H15\" | grep -v '\-\-' | awk '{print $2, $3, $4}'"
    #command = "grep norm: " + fileN + "| awk '{print $2}'"
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    #  I am reading a list of lists(triplets) in bytes. I need to separate them

    outputO = np.array([list(map(float, x.split(b' '))) for x in output.split(b'\n')
        if x != b''])
    outputDivided = list(chunksOf(outputO,natoms))

    # I need to fill the NAC matrix with the elements I have...
    for index in range(len(outputDivided)):
        # indexes of out of diagonal terms
        ind1 = np.triu_indices(nstates,1)[0][index]
        ind2 = np.triu_indices(nstates,1)[1][index]
        emptyMat[ind1, ind2] = outputDivided[index]
        # tricky NAC, you're antisymmetric
        emptyMat[ind2, ind1] = -outputDivided[index]
    return(emptyMat)


def unpackThingsFromParallel(listOfval):
    '''
    this unpacks the result list of tuples from the parallel execution
    listOfval :: (String,Boolean) <- all the messages and values packed in
    tuples
    '''
    log = '\n'.join([x[0] for x in listOfval])
    length = len(listOfval)
    trues = [x[1] for x in listOfval].count(True)
    return(log,trues,length-trues)


def correctThis(fileN,oneDarray,outputF,cutAt,first=None):
    '''
    This is the corrector. Go go go
    fileN :: String <- the path of the h5 file
    oneDarray :: np.array(NSTATES) <- this is the 1D vector that tells us how
                                      sign changed in LAST CALCULATION
    outputF :: String <- the path of the output folder
    cutAt :: Int <- wanna less states?
    '''
    first = first or False
    dataToGet = ['OVERLAP', 'DIPOLES', 'NAC']
    [overlapsM, dipolesAll, nacAll] = retrieve_hdf5_data(fileN, dataToGet)
    if first:
        (_, nstates, _) = dipolesAll.shape
        overlapsAll = np.identity(nstates)
    else:
        (nstates, _ ) = overlapsM.shape
        overlapsAll = overlapsM # leave this here for now

    # let's cut something
    dipoles = dipolesAll[:, :cutAt, :cutAt]
    overlaps = overlapsAll[:cutAt, :cutAt]
    print(nacAll.shape)
    nacs = nacAll[:cutAt, :cutAt]

    #correctionArray1DABS = createOneAndZeroABS(overlaps, oneDarray)
    correctionArray1DABS = compressColumnOverlap(overlaps, oneDarray)
    correctionMatrix = createTabellineFromArray(correctionArray1DABS)
    new_dipoles = dipoles * correctionMatrix
    # here I use the fact that correctionMatrix is ALWAYS 2d
    # so I loop over the it
    new_nacs = np.empty_like(nacs)
    for i in range(nstates):
        for j in range(nstates):
            new_nacs[i,j] = nacs[i,j] * correctionMatrix[i,j]


    print('\n')
    print('This is overlap:')
    printMatrix2D(overlaps,2)
    print('\n\n')
    print('oneD -> {}\ncorr -> {}'.format(oneDarray,correctionArray1DABS))
    print('\n\n')
    print('this is correction Matrix')
    printMatrix2D(correctionMatrix,2)
    print('\n\n')
    print('These are the old dipoles:')
    printMatrix2D(dipoles[0],2)
    print('\n\n')
    print('These are the new dipoles:')
    printMatrix2D(new_dipoles[0],2)
    print('\n\n')
    print('These are the old NACS:')
    printMatrix2D(nacs[:,:,9,1],2)
    print('\n\n')
    print('These are the new NACS:')
    printMatrix2D(new_nacs[:,:,9,1],2)


    # file handling
    corrFN = os.path.basename(os.path.splitext(fileN)[0] + 'corrected.h5')
    corrFNO = os.path.join(outputF,corrFN)
    allValues = readWholeH5toDict(fileN)
    allValues['DIPOLES'] = new_dipoles
    allValues['NAC'] = new_nacs
    writeH5fileDict(corrFNO,allValues)
    print('\n\nfile {} written'.format(corrFNO))


    return correctionArray1DABS


def directionRead(folderO,folderE):
    '''
    I NEED TO CHECK THAT THESE CALCULATIONS ARE CONSECUTIVE !!!
    the filter function is taking out falses, without checking anything !!!
    watchout = list(filter(os.path.isfile,mainLine))
    '''
    phis = ['P000-000', 'P001-000', 'P002-000', 'P003-000', 'P004-000',
      'P005-000', 'P006-000', 'P007-000']
    gammas = ['P010-000', 'P010-714', 'P011-429', 'P012-143', 'P012-857',
      'P013-571', 'P014-286', 'P015-000', 'P015-714', 'P016-429', 'P017-143',
      'P017-857', 'P018-571', 'P019-286', 'P020-000']
    thetas = ['P120-000', 'P119-184', 'P118-367', 'P117-551', 'P116-735',
      'P115-918', 'P115-102', 'P114-286', 'P113-469', 'P112-653', 'P111-837',
      'P111-020', 'P110-204', 'P109-388', 'P108-571', 'P107-755', 'P106-939',
      'P106-122', 'P105-306', 'P104-490', 'P103-673', 'P102-857', 'P102-041',
      'P101-224', 'P100-408', 'P099-592', 'P098-776', 'P097-959', 'P097-143',
      'P096-327', 'P095-510', 'P094-694', 'P093-878', 'P093-061', 'P092-245',
      'P091-429', 'P090-612', 'P089-796', 'P088-980', 'P088-163', 'P087-347',
      'P086-531', 'P085-714', 'P084-898', 'P084-082', 'P083-265', 'P082-449',
      'P081-633', 'P080-816', 'P080-000']
    rootName = os.path.join(folderO,'zNorbornadiene_')
    mainLine = [ rootName + 'P000-000_P010-000_' + theta + '.all.h5' for theta
            in thetas ]
    # this filter is NOT exactly what you want
    watchout = list(filter(os.path.isfile,mainLine))
    cutAt = 14
    newsign = np.ones(cutAt)
    for i in range(len(watchout)):
    #for i in range(3):
        if i == 0:
            print('\n\n\n----------THIS IS INITIAL -> cut at {}:\n'.format(cutAt))
            correctThis(watchout[i],newsign,folderE,cutAt,True)
        else:
            print('\n\n\n----------THIS IS {} -> cut at {}:\n'.format(i,cutAt))
            newsign = correctThis(watchout[i],newsign,
                    folderE,cutAt)


def compressColumnOverlap(mat, oneDarray):
    '''
    mat :: np.array(X,Y) <- an overlap matrix
    given a matrix with overlaps this will return an array of +1 or -1.
    This will determine sign changes for this step.
    '''
    axis = 0
    amax = mat.max(axis)
    amin = mat.min(axis)
    result = np.where(-amin > amax, -1., 1.)
    return (result*oneDarray)

def createOneAndZeroABS(mat, oneDarray):
    '''
    mat :: np.array(X,Y) <- an overlap matrix
    given a matrix with overlaps this will return a matrix with 0 and 1 and -1
    This will determine sign changes for this step.

    Somehow this function is broken

    '''
    (dimension,_) = mat[:].shape
    new = np.zeros_like(mat)
    for i in np.arange(dimension):
        amax = mat[i].max()
        amin = mat[i].min()
        result = np.where(-amin > amax, amin, amax)
        [due] = np.argwhere(np.isin(mat[i],result))[0]
        print(amax,amin,result,due)
        if result > 0:
            new [i,due] = 1
        else:
            new [i,due] = -1
    print('mult {} for:\n{}'.format(oneDarray, new))
    return (oneDarray @ new)


single_inputs = namedtuple("single_input", ("direction","glob","outF","proc"))

stringOutput1 = '''
{}

finished -> {}
problems -> {}
'''


def main():
    '''
    from a grid frolder to a folder ready to be propagated
    '''
    o_inputs = single_inputs("","","",1)
    inp = read_single_arguments(o_inputs)
    if inp.direction == "":
        folders = npArrayOfFiles(inp.glob)
        fold = folders[:] # this does nothing, but I use it to try less files at the time
        pool = multiprocessing.Pool(processes = inp.proc)
        resultP = pool.map(createOutputFile, zip(fold, repeat(inp.outF)))
        (logAll,finished,problems) = unpackThingsFromParallel(resultP)
        results = stringOutput1.format(logAll, finished, problems)

        with open('report','w') as f:
            f.write(results)
        print(results)
    else:
        directionRead('/home/alessio/Desktop/a-3dScanSashaSupport/f-outputs',
        '/home/alessio/Desktop/a-3dScanSashaSupport/g-outsCorrected')


if __name__ == "__main__":
    main()




#    mat = np.array([[1.00e+00,-6.51e-03,2.93e-06,-1.35e-07],
#    [-4.93e-03, -1.00e+00, -3.68e-05, 3.71e-06],
#    [ 1.16e-05, 1.83e-06, -3.90e-06, -9.32e-01],
#    [ 1.93e-05, -2.95e-05, 9.47e-01, -1.30e-04]])
#    oneDarray = np.array([-1.,-1.,-1.,-1.])
#    re = compressColumnOverlap(mat,oneDarray)
#    re2 = compressColumnOverlap2(mat,oneDarray)
#    print('{}\nenter    {}\nresult 1 {}\nresult 2 {}'.format(mat,oneDarray,re,re2))
#
