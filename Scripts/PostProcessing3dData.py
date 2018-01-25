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
                       writeH5fileDict, readWholeH5toDict)


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
    #h5dipole = root + '.transDip.h5'
    h5rassi = root + '.rassi.h5'
    h5out = root + '.out'
    a_exist = all([ os.path.isfile(f) for f in
        [h5rassi,h5rasscf,h5out]])
    if a_exist:
        log = proj + ': all files present'
        boolean = True
        [geom,aType,ciVect,ener] = retrieve_hdf5_data(h5rasscf,
                                   ['CENTER_COORDINATES',
                                    'CENTER_LABELS',
                                    'CI_VECTORS',
                                    'ROOT_ENERGIES'])
        #[dipoles, transDen] = retrieve_hdf5_data(h5dipole,
        #                      ['PROPERTIES',
        #                       'SFS_TRANSITION_DENSITIES'])
        [dipoles, overlap] = retrieve_hdf5_data(h5rassi,
                           ['PROPERTIES','ORIGINAL_OVERLAPS'])
        NAC = parseNAC(h5out)
        outfile = oroot + '.all.h5'

        outTuple = [('CENTER_COORDINATES', geom),
                    ('CENTER_LABELS', aType),
                    ('CI_VECTORS', ciVect),
                    ('ROOT_ENERGIES', ener),
                    ('DIPOLES', dipoles[0:3]),
                    #('TRANDENS',transDen),
                    ('OVERLAP', overlap),
                    ('NAC', NAC)]

        try:
           writeH5file(outfile,outTuple)
        except "Unable to open file":
           print(outfile)
        log += ' -> ' + str(NAC.size)
        #print('File ' + outfile + ' written.')
    else:
        log = proj + ': this calculation is not completed'
        boolean = False
    return(log,boolean)


def parseNAC(fileN):
    '''
    I use bash here to quickly get out NAC values
    fileOut :: filePath <- the output of molcas
    '''
    import subprocess
    command = "grep norm: " + fileN + "| awk '{print $2}'"
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    outputO = np.array([float(x) for x in output.split(b'\n') if x != b'' ])
    return(outputO)


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
        ####     serial Function
        #for i in fold:
        #    (log,boolean) = createOutputFile(i,inp.outF)
        #    logAll += '\n' + log
        #    if boolean:
        #        finished += 1
        #    else:
        #        problems += 1
        #### go parallel here
        pool = multiprocessing.Pool(processes = inp.proc)
        resultP = pool.map(createOutputFile, zip(fold, repeat(inp.outF)))
        (logAll,finished,problems) = unpackThingsFromParallel(resultP)
        results = stringOutput1.format(logAll,finished,problems)

        with open('report','w') as f:
            f.write(results)
        print(results)
    else:
        directionRead('/home/alessio/Desktop/a-3dScanSashaSupport/f-outputs',
        '/home/alessio/Desktop/a-3dScanSashaSupport/g-outsCorrected')


def correctThis(fileN,oneDarray,outputF,cutAt):
    '''
    This is the corrector. Go go go
    fileN :: String <- the path of the h5 file
    oneDarray :: np.array(NSTATES) <- this is the 1D vector that tells us how
                                      sign changed in LAST CALCULATION
    outputF :: String <- the path of the output folder
    '''
    dataToGet = ['OVERLAP', 'DIPOLES']
    [overlapsM, dipolesM] = retrieve_hdf5_data(fileN, dataToGet)
    (dim, _ ) = overlapsM.shape
    nstates = dim // 2
    overlapsAll = overlapsM[nstates:,:nstates]
    dipolesAll = dipolesM[:,nstates:,nstates:]

    # let's cut something
    overlaps = overlapsAll[:cutAt, :cutAt]
    dipoles = dipolesAll[:, :cutAt, :cutAt]

    correctionArray1DABS = createOneAndZeroABS(overlaps, oneDarray)
    correctionArray1DABS2 = compressColumnOverlap(overlaps, oneDarray)
    correctionMatrix = createTabellineFromArray(correctionArray1DABS)
    new_dipoles = dipoles * correctionMatrix

    # file handling
    corrFN = os.path.basename(os.path.splitext(fileN)[0] + 'corrected.h5')
    corrFNO = os.path.join(outputF,corrFN)
    allValues = readWholeH5toDict(fileN)
    allValues['DIPOLES'] = new_dipoles
    writeH5fileDict(corrFNO,allValues)
    print('file {} written'.format(corrFNO))
    print('\n\n')
    print('This is overlap:')
    printMatrix2D(overlaps,2)
    print('\n\n')
    print('this is correction Matrix')
    printMatrix2D(correctionMatrix,2)
    print('\n\n')
    print('These are the old dipoles:')
    printMatrix2D(dipoles[0],2)
    print('\n\n')
    print('These are the new dipoles:')
    printMatrix2D(new_dipoles[0],2)
    return correctionArray1DABS


def directionRead(folderO,folderE):
    '''
    I NEED TO CHECK THAT THESE CALCULATIONS ARE CONSECUTIVE !!!
    filter is taking out falses, without checking anything !!!
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
    newsign = np.ones(cutAt)   # problem BOUND for states...
    for i in range(len(watchout)):
        if i != 0:
            print('\n\n\n----------THIS IS {} -> cut at {}:\n'.format(i,cutAt))
            newsign = correctThis(watchout[i],newsign,
                    folderE,cutAt)
            dipolesAll = retrieve_hdf5_data(watchout[i], 'DIPOLES')


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
    it is shitty, but we need to see if this works or not
    '''
    axis = 0
    amax = mat.max(axis)
    amin = mat.min(axis)
    result = np.where(-amin > amax, amin, amax)
    new = np.zeros_like(mat)
    for i in result:
        [uno,due] = np.argwhere(np.isin(mat,i))[0]
        if i > 0:
            new[uno,due] = 1
        else:
            new[uno,due] = -1
    return (oneDarray @ new)

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
