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
                       npArrayOfFiles, printMatrix2D, createTabellineFromArray)


def read_single_arguments(single_inputs):
    '''
     This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-o", "--outputFolder",
                        dest="o",
                        type=str,
                        required=True,
                        help="The location of the output folder")
    parser.add_argument("-s", "--scanFolder",
                        dest="s",
                        type=str,
                        required=True,
                        help="The location of the Grid folder")
    parser.add_argument("-p", "--parallel",
                        dest="p",
                        type=int,
                        help="number of processors if you want it parallel")

    args = parser.parse_args()
    if args.o != None:
        single_inputs = single_inputs._replace(outF=args.o)
    if args.s != None:
        single_inputs = single_inputs._replace(glob=args.s)
    if args.p != None:
        single_inputs = single_inputs._replace(proc=args.p)

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
        [h5rassi,h5rasscf,h5dipole,h5out]])
    if a_exist:
        log = proj + ': all files present'
        boolean = True
        [geom,aType,ciVect,ener] = retrieve_hdf5_data(h5rasscf,
                                   ['CENTER_COORDINATES',
                                    'CENTER_LABELS',
                                    'CI_VECTORS',
                                    'ROOT_ENERGIES'])
        [dipoles, transDen] = retrieve_hdf5_data(h5dipole,
                              ['PROPERTIES',
                               'SFS_TRANSITION_DENSITIES'])
        overlap = retrieve_hdf5_data(h5rassi,'ORIGINAL_OVERLAPS')
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


single_inputs = namedtuple("single_input", ("glob","outF","proc"))


def main():
    '''
    from a grid frolder to a folder ready to be propagated
    '''
    o_inputs = single_inputs("","",1)
    inp = read_single_arguments(o_inputs)
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
    results = '''
{}

finished -> {}
problems -> {}
'''.format(logAll,finished,problems)

    with open('report','w') as f:
        f.write(results)
    print(results)


def compressColumnOverlap(mat):
    '''
    mat :: np.array(X,Y) <- an overlap matrix
    given a matrix with overlaps this will return an array of +1 or -1.
    This will determine sign changes for this step.
    '''
    axis = 0
    amax = mat.max(axis)
    amin = mat.min(axis)
    return np.where(-amin > amax, -1, 1)


def correctThisfromThat(file1,file2):
    '''
    This is the corrector. Go go go
    file1,file2 :: String <- the path of the two h5 files
    '''
    overlapsM = retrieve_hdf5_data(file2, 'OVERLAP')
    (dim, _ ) = overlapsM.shape
    nstates = dim // 2
    overlaps = overlapsM[nstates:,:nstates]
    printMatrix2D(overlaps,2)
    arrayOneD = compressColumnOverlap(overlaps)
    correctionMatrix = createTabellineFromArray(arrayOneD)

    print(file1,file2)

def directionRead(folder):
    '''
    I NEED TO CHECK THAT THESE CALCULATIONS ARE CONSECUTIVE !!!
    filter is taking aout false, without checking anything !!!
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
    rootName = os.path.join(folder,'zNorbornadiene_')
    mainLine = [ rootName + 'P000-000_P010-000_' + theta + '.all.h5' for theta
            in thetas ]
    watchout = list(filter(os.path.isfile,mainLine))
    correctThisfromThat(watchout[0],watchout[1])

if __name__ == "__main__":
#    main()
    directionRead('/home/alessio/Desktop/a-3dScanSashaSupport/f-outputs')

