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
from shutil import copy2

from quantumpropagator import (retrieve_hdf5_data, writeH5file,
                       npArrayOfFiles, printMatrix2D, createTabellineFromArray,
                       writeH5fileDict, readWholeH5toDict, chunksOf, err, good, warning,
                       printDict, stringTransformation3d, calc_g_G, readDirectionFile)

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
                        "Folder of h5 files\n"
                        "Folder of outputs\n")
    parser.add_argument("-r", "--refine",
                        dest="r",
                        nargs='+',
                        help="refiner, takes two arguments:\n"
                        "Folder of not refined h5\n"
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
    if args.r != None:
        single_inputs = single_inputs._replace(refine=args.r)

    return single_inputs


def createOutputFile(tupleI):
    '''
    given a projectname, creates single outFile (not yet corrected)
    tupleI :: (String,String) <- is composed by (fold,outFol)
    fold :: String <- the folder of a single calculation
    outFol :: String <- the folder where the output is collected
    '''
    (fold,outFol) = tupleI
    #print('doing ' + fold)
    folder = os.path.dirname(fold)
    proj = os.path.basename(fold)
    root = folder +'/' + proj + '/' + proj
    oroot = os.path.join(outFol,proj)
    h5rasscf = root + '.rasscf.h5'
    h5dipole = root + '.transDip.h5'
    h5rassi = root + '.rassi.h5'
    h5out = root + '.out'
    a_exist = all([ os.path.isfile(f) for f in
        #[h5rassi,h5rasscf,h5dipole]])   Soon we will not need out anymore
        [h5rassi,h5rasscf,h5out,h5dipole]])
    if a_exist:
        log = proj + ': all files present'

        boolean = True

        [geom,aType,ener] = retrieve_hdf5_data(h5rasscf,
                                   ['CENTER_COORDINATES',
                                    'CENTER_LABELS',
                                    'ROOT_ENERGIES'])

        [dipoles, transDen] = retrieve_hdf5_data(h5dipole,
                              ['SFS_EDIPMOM',
                               'SFS_TRANSITION_DENSITIES'])
        [overlap] = retrieve_hdf5_data(h5rassi,
                           ['ORIGINAL_OVERLAPS'])
        nstates = ener.size
        nstatesNAC = 8 # states for nac are actually 8
        natoms = aType.size
        # I want to save only the low left corner of overlap
        if True:
            NAC = parseNAC(h5out,nstatesNAC,natoms)
        else:
            warning("NAC PARSING TURNED OFF")
            NAC = np.zeros((nstatesNAC,nstatesNAC,natoms,3))

        outfile = oroot + '.all.h5'
        outTuple = [('CENTER_COORDINATES', geom),
                    ('CENTER_LABELS', aType),
                    ('ROOT_ENERGIES', ener),
                    ('DIPOLES', dipoles),
                    #('CI_VECTORS', ciVect),
                    #('TRANDENS',transDen),
                    ('OVERLAP', overlap[nstates:,:nstates]),
                    ('NAC', NAC)]
        #print(overlap[nstates:,:nstates][:3,:3])
        #print(overlap)
        #try:
        #    if (overlap[nstates:,:nstates][:3,:3] == np.zeros((3,3))).all():
        #        log += 'dioK {}\n'.format(outfile)
        #except:
        #    err(oroot)

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

    # this parser need to filter out the b'' objects that forms from the function split.
    # also, there is a problem with numbers not separated in molcas outputs, 
    # that is why we replace '-' # with ' -'

    outputO = np.array([ getThreeNumbers(x) for x in output.split(b'\n') if x != b''])
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

def getThreeNumbers(bytestr):
    '''
    this function exists because Molcas does not always give NAC in a suitable way
    sometimes the grepped lines are like this
    9736.8701781410642.89824311 4710.70644771
    ************** 8777.30913244 474.27759987
    so I am using a few tricks to convert these values in something big, but still triplets of numbers
    '''
    res = (bytestr.replace(b'-',b' -')).split(b' ')
    noEmpty = [ x for x in res if x != b'' ]
    if len(noEmpty) < 3:
        ris = [ 9999, 9999, 9999 ]
    else:
        ris = []
        for elem in noEmpty:
            try:
                val = float(elem)
                ris.append(val)
            except ValueError:
                ris.append(9999)
    return(ris)


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


def refineStuffs(folderO,folderE,fn1,fn2):
    '''
    this is the old brother of correctorFromDirection
    I am trying to get a refinement and catch points that suddenly changes sign.
    I am lucky that theta and gamma when Phi=0 seems good, so I try to force out the changes
    in Phi.
    '''
    phis1,gammas1,thetas1 = readDirectionFile(fn1)
    phis2,gammas2,thetas2 = readDirectionFile(fn2)
    rootNameO = os.path.join(folderO,'zNorbornadiene')
    rootNameE = os.path.join(folderE,'zNorbornadiene')
    flat_range_Gamma = gammas1[::-1]+gammas2[1:]
    flat_range_Theta = (thetas1[::-1]+thetas2[1:])[::-1]

    dataToGet = ['DIPOLES', 'NAC']

    for gamLab in flat_range_Gamma:
        for theLab in flat_range_Theta:
            AbsCorr = np.ones((3,8,8))
            for p,phiLab in enumerate(phis1[:-1]):
                elemT = '{}_{}_{}'.format(phiLab,gamLab,theLab)
                elemN = '{}_{}_{}'.format(phis1[p+1],gamLab,theLab)
                THIS = '{}_{}.corrected.h5'.format(rootNameO,elemT)
                NEXT = '{}_{}.corrected.h5'.format(rootNameO,elemN)
                OUTN = '{}_{}.refined.h5'.format(rootNameE,elemN)
                if p==0:
                    OUTT = '{}_{}.refined.h5'.format(rootNameE,elemT)
                    copy2(THIS,OUTT)
                [dipolesAllT, nacAllT] = retrieve_hdf5_data(THIS, dataToGet)
                [dipolesAllN, nacAllN] = retrieve_hdf5_data(NEXT, dataToGet)
                _,nstates,_ = dipolesAllT.shape
                # loop on lower triangule
                for cart in range(3):
                    for i in range(nstates):
                        for j in range(i):
                            uno = dipolesAllT[cart,i,j]
                            due = dipolesAllN[cart,i,j]
                            sign1 = np.sign(uno)
                            sign2 = np.sign(due)
                            if abs(uno) > 10e-2 and abs(due) > 10e-2 and sign1 != sign2:
                                AbsCorr[cart,i,j] = -AbsCorr[cart,i,j]
                                print('change in {} -> {} ({},{},{})'.format(elemT,elemN,cart,i,j))
                            dipolesAllN[cart,i,j] = due * AbsCorr[cart,i,j]
                            dipolesAllN[cart,j,i] = due * AbsCorr[cart,i,j]


                # file handling
                allValues = readWholeH5toDict(NEXT)
                allValues['DIPOLES'] = dipolesAllN
                #allValues['NAC'] = new_nacs
                writeH5fileDict(OUTN,allValues)





def correctorFromDirection(folderO,folderE,fn1,fn2):
    '''
    This function is the corrector that follows the direction files...
    suuuuper problem bound
    '''
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
    cutAt = 8
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
            print('\n\n----------THIS IS {} from {} -> cut at {}:\n'.format(key,value,cutAt))
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
    dataToGet = ['ROOT_ENERGIES','OVERLAP', 'DIPOLES', 'NAC']
    fileN = rootNameO + elem + '.all.h5'

    # I add a string LOL in front of elem to make it equal to a normal file name, but elem here
    # is just the three labels (small dirty fix)
    # stringTransformation3d changes the UNITS of the labels, it is not anymore a simple tofloat
    phiA,_,gammaA,_,thetA,_ = stringTransformation3d("LOL_" + elem)

    [enerAll, overlapsM, dipolesAll, nacAll] = retrieve_hdf5_data(fileN, dataToGet)
    if first:
        (_, nstates, _) = dipolesAll.shape
        overlapsAll = np.identity(nstates)
    else:
        (nstates, _ ) = overlapsM.shape
        overlapsAll = overlapsM # leave this here for now

    # let's cut something
    energies = enerAll[:cutAt]
    dipoles = dipolesAll[:, :cutAt, :cutAt]
    overlaps = overlapsAll[:cutAt, :cutAt]
    nacs = nacAll[:cutAt, :cutAt]

    correctionArray1DABS, overlap_one_zero = createOneAndZero(overlaps, oneDarray)
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
    allValues['OVERLAPONEZERO'] = overlap_one_zero
    allValues['KINETIC_COEFFICIENTS'] = calc_g_G(phiA,gammaA,thetA)
    allValues['ROOT_ENERGIES'] = energies
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
    newcorrVector = oneDarray @ newMat.T
    if not booltest:
        print('This is an overlap matrix that you should check')
        print('entrance matrix')
        printMatrix2D(mat)
        print('entrance array:')
        print(oneDarray)
        print('exit matrix')
        printMatrix2D(newMat)
        print('exit array:')
        print(newcorrVector)
    return (newcorrVector,newMat)


single_inputs = namedtuple("single_input", ("direction","refine","glob","outF","proc"))

stringOutput1 = '''
{}

finished -> {}
problems -> {}

file report written
'''


def main():
    '''
    from a grid frolder to a folder ready to be propagated
    '''
    fn1 = '/home/alessio/Desktop/a-3dScanSashaSupport/o-FinerProjectWithNAC/directions1'
    fn2 = '/home/alessio/Desktop/a-3dScanSashaSupport/o-FinerProjectWithNAC/directions2'
    o_inputs = single_inputs("","","","",1)
    inp = read_single_arguments(o_inputs)
    if inp.direction == "" and inp.refine == "":
        folders = npArrayOfFiles(inp.glob)
        fold = folders[:] # this does nothing, but I use it to try less files at the time
        pool = multiprocessing.Pool(processes = inp.proc)
        resultP = pool.map(createOutputFile, zip(fold, repeat(inp.outF)))
        (logAll,finished,problems) = unpackThingsFromParallel(resultP)
        results = stringOutput1.format(logAll, finished, problems)

        with open('report','w') as f:
            f.write(results)
        print(results)
    elif inp.refine != "":
        # here the code for the refining thing. big stuff...
        #print(inp.refine[0],inp.refine[1],fn1,fn2)
        refineStuffs(inp.refine[0],inp.refine[1],fn1,fn2)
    else:
        correctorFromDirection(inp.direction[0],inp.direction[1],fn1,fn2)

if __name__ == "__main__":
    main()

