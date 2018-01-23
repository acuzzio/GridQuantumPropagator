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
                       npArrayOfFiles)


def read_single_arguments(single_inputs):
    '''
     This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-o", "--outputFolder",
                        dest="o",
                        type=str,
                        required=True,
                        help="The location of the Grid folder")
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


#def createOutputFile(fold,outFol):
def createOutputFile(tupleI):
    '''
    given a projectname, creates single outFile (not yet corrected)
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
#                    ('TRANDENS',transDen),
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


if __name__ == "__main__":
        main()

