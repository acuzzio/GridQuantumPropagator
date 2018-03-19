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
                       writeH5fileDict, readWholeH5toDict, chunksOf, err, good,
                       printDict)


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


def makeCubeGraph(phis,gammas,thetas):
    '''
    This supremely problem bound function, will create a dictionary of
    "directions" to run analysis functions across the scan in the right order
    '''
    graph = {}

    gammaI = gammas[0]
    phiI = phis[0]
    thetaL = thetas[-1]
    gammaL = gammas[-1]
    phiL = phis[-1]
    for k in range(len(thetas)):
        for j in range(len(gammas)):
            for i in range(len(phis)):
                name = '_'.join((phis[i],gammas[j],thetas[k]))
                graph[name] = []
                if phis[i] == phiI and gammas[j] == gammaI and thetas[k] != thetaL:
                   succt = '_'.join((phis[i],gammas[j],thetas[k+1]))
                   graph[name].append(succt)
                if phis[i] == phiI and gammas[j] != gammaL:
                   succg = '_'.join((phis[i],gammas[j+1],thetas[k]))
                   graph[name].append(succg)
                if phis[i] != phiL:
                   succp = '_'.join((phis[i+1],gammas[j],thetas[k]))
                   graph[name].append(succp)
    # return the INVERSE dictionary, so I know from where I should take the
    # correction vector
    reverseGraph = dict((v,k) for k in graph for v in graph[k])
    first = '_'.join((phis[0],gammas[0],thetas[0]))
    return(graph,reverseGraph,first)


def readDirectionFile(fn):
    '''
    fn :: filePath
    '''
    with open(fn,'r') as f:
        f.readline()
        phis = f.readline()
        f.readline()
        f.readline()
        gammas = f.readline()
        f.readline()
        f.readline()
        thetas = f.readline()
    return(phis.rstrip().split(' '),gammas.rstrip().split(' '), thetas.rstrip().split(' '))

def directionRead(folderO,folderE):
    '''
    This function is the corrector that follows the direction files...
    suuuuper problem bound
    '''
    fn1 = '/home/alessio/Desktop/a-3dScanSashaSupport/m-biggerScanEnergies2ways/directions1'
    fn2 = '/home/alessio/Desktop/a-3dScanSashaSupport/m-biggerScanEnergies2ways/directions2'
    phis1,gammas1,thetas1 = readDirectionFile(fn1)
    phis2,gammas2,thetas2 = readDirectionFile(fn2)
    #phis = phis[0:3]
    #gammas = gammas[0:3]
    #thetas = thetas[0:3]
    rootNameO = os.path.join(folderO,'zNorbornadiene_')
    rootNameE = os.path.join(folderE,'zNorbornadiene_')
    graph1,revgraph1,first = makeCubeGraph(phis1,gammas1,thetas1)
    graph2,revgraph2,_ = makeCubeGraph(phis2,gammas1,thetas1)
    graph3,revgraph3,_ = makeCubeGraph(phis1,gammas2,thetas1)
    graph4,revgraph4,_ = makeCubeGraph(phis2,gammas2,thetas1)
    graph5,revgraph5,_ = makeCubeGraph(phis1,gammas1,thetas2)
    graph6,revgraph6,_ = makeCubeGraph(phis2,gammas1,thetas2)
    graph7,revgraph7,_ = makeCubeGraph(phis1,gammas2,thetas2)
    graph8,revgraph8,_ = makeCubeGraph(phis2,gammas2,thetas2)
    cutAt = 14
    # correct first point here - True means "I am the first"
    print('\n\n----------THIS IS INITIAL -> cut at {}:\n'.format(cutAt))
    newsign = np.ones(cutAt)
    correctThis(first,newsign,rootNameE,rootNameO,cutAt,True)
    # correct the other here key is file to be corrected VALUE the one where to
    # take the correction
    #print("{}\n{}\n{}".format(phis,gammas,thetas))
    #print(' ')
    #printDict(graph)
    #print(' ')
    #printDict(revgraph)
    revgraphSum = {**revgraph8,
                   **revgraph7,
                   **revgraph6,
                   **revgraph5,
                   **revgraph4,
                   **revgraph3,
                   **revgraph2,
                   **revgraph1}
    #print(len(revgraphSum))
    for key, value in revgraphSum.items():
        fnIn = rootNameO + key + '.all.h5'
        if os.path.isfile(fnIn):
            print('\n\n----------THIS IS {} -> cut at {}:\n'.format(key,cutAt))
            fnE = rootNameE + value + '.corrected.h5'
            newsign = retrieve_hdf5_data(fnE,'ABS_CORRECTOR')
            correctThis(key,newsign,rootNameE,rootNameO,cutAt)
    good('Hey, you are using an hardcoded direction file')


def correctThis(elem,oneDarray,rootNameE,rootNameO,cutAt,first=None):
    '''
    This is the corrector. Go go go
    elem :: String <- the label of the h5 file
    oneDarray :: np.array(NSTATES) <- this is the 1D vector that tells us how
                                      sign changed in LAST CALCULATION
    '''
    first = first or False
    dataToGet = ['OVERLAP', 'DIPOLES', 'NAC']
    fileN = rootNameO + elem + '.all.h5'
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
    nacs = nacAll[:cutAt, :cutAt]

    correctionArray1DABS = createOneAndZero(overlaps, oneDarray)
    correctionMatrix = createTabellineFromArray(correctionArray1DABS)
    new_dipoles = dipoles * correctionMatrix
    # here I use the fact that correctionMatrix is ALWAYS 2d
    # so I loop over the it
    new_nacs = np.empty_like(nacs)
    for i in range(cutAt):
        for j in range(cutAt):
            new_nacs[i,j] = nacs[i,j] * correctionMatrix[i,j]


    print('\n')
    print('This is overlap:')
    printMatrix2D(overlaps,2)
    print('\n\n')
    print('from Previous\n {}\neffective correction:\n {}'.format(oneDarray,correctionArray1DABS))
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
    corrFNO = rootNameE + elem + '.corrected.h5'
    allValues = readWholeH5toDict(fileN)
    allValues['DIPOLES'] = new_dipoles
    allValues['NAC'] = new_nacs
    allValues['ABS_CORRECTOR'] = correctionArray1DABS
    writeH5fileDict(corrFNO,allValues)
    print('\n\nfile {} written'.format(corrFNO))


def createOneAndZero(mat, oneDarray):
    '''
    mat :: np.array(X,Y) <- an overlap matrix
    given a matrix with overlaps this will return an array with 1 and -1
    This will determine sign changes for this step.

    This function is quite convoluted. Rows are new states, so I go along rows
    and seek for the absolute maximum value. if this is negative I put a -1, or
    a +1 if it is positive, then I take out the state I just assigned (so it
    cannot be assigned again). This is to avoid overlap matrices with double 1
    or no 1 at all (they happen).
    '''

    # Maximum implementation
    newMat = np.empty_like(mat)
    ind = 0
    taken = []
    for line in mat:
        i = np.copy(line)
        i[taken] = 0
        maxL,minL = (i.max(),i.min())
        if -minL > maxL:
            newMat[ind] = np.where(i==minL,-1,0)
            take = np.argwhere(i==minL)[0][0]
            taken += [take]
        else:
            newMat[ind] = np.where(i==maxL, 1,0)
            take = np.argwhere(i==maxL)[0][0]
            taken += [take]
        ind+=1

    booltest = np.all(np.count_nonzero(newMat,axis = 0) == 1)
    # I check if all columns and rows have exactly one one.
    if not booltest:
        print('there is an overlap matrix that you should check')
        printMatrix2D(newMat)
    newcorrVector = oneDarray @ newMat.T
    return (newcorrVector)


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
        directionRead(inp.direction[0],inp.direction[1])

if __name__ == "__main__":
    main()
    #phis = ['P000-000', 'P001-000', 'P002-000', 'P003-000', 'P004-000',
    #  'P005-000', 'P006-000', 'P007-000']
    #gammas = ['P010-000', 'P010-714', 'P011-429', 'P012-143', 'P012-857',
    #  'P013-571', 'P014-286', 'P015-000', 'P015-714', 'P016-429', 'P017-143',
    #  'P017-857', 'P018-571', 'P019-286', 'P020-000']
    #thetas = ['P120-000', 'P119-184', 'P118-367', 'P117-551', 'P116-735',
    #  'P115-918', 'P115-102', 'P114-286', 'P113-469', 'P112-653', 'P111-837',
    #  'P111-020', 'P110-204', 'P109-388', 'P108-571', 'P107-755', 'P106-939',
    #  'P106-122', 'P105-306', 'P104-490', 'P103-673', 'P102-857', 'P102-041',
    #  'P101-224', 'P100-408', 'P099-592', 'P098-776', 'P097-959', 'P097-143',
    #  'P096-327', 'P095-510', 'P094-694', 'P093-878', 'P093-061', 'P092-245',
    #  'P091-429', 'P090-612', 'P089-796', 'P088-980', 'P088-163', 'P087-347',
    #  'P086-531', 'P085-714', 'P084-898', 'P084-082', 'P083-265', 'P082-449',
    #  'P081-633', 'P080-816', 'P080-000']
    #phis = phis[0:3]
    #gammas = gammas[0:3]
    #thetas = thetas[0:3]
    #graph,revG,first = makeCubeGraph(phis,gammas,thetas)
    #print(first)
    #print('')
    #printDict(graph)
    #print('')
    #printDict(revG)
    #print('')




#    mat = np.array([[1.00e+00,-6.51e-03,2.93e-06,-1.35e-07],
#    [-4.93e-03, -1.00e+00, -3.68e-05, 3.71e-06],
#    [ 1.16e-05, 1.83e-06, -3.90e-06, -9.32e-01],
#    [ 1.93e-05, -2.95e-05, 9.47e-01, -1.30e-04]])
#    oneDarray = np.array([-1.,-1.,-1.,-1.])
#    re = compressColumnOverlap(mat,oneDarray)
#    re2 = createOneAndZeroABS(mat,oneDarray)
#    re3 = createOneAndZeroABS2(mat,oneDarray)
#    print('''{}
#    enter    {}
#    result 1 {}
#    result 2 {}
#    result 3 {}'''.format(mat,oneDarray,re,re2,re3))

