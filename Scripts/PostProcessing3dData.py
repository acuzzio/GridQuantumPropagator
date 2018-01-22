'''
This scripts takes care of postprocessing and reorganizing/correcting
data after Molcas scan...
'''

from collections import namedtuple
from argparse import ArgumentParser

import glob

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
    args = parser.parse_args()
    if args.o != None:
        single_inputs = single_inputs._replace(outF=args.o)
    if args.s != None:
        single_inputs = single_inputs._replace(glob=args.s)
    return single_inputs

def parseNAC(fileOut):
    '''
    I use bash here to quickly get out NAC values
    fileOut :: filePath <- the output of molcas
    '''
    return(np.empty(10))


def createOutputFile(fold,outFol):
    '''
    given a projectname, creates single outFile (not yet corrected)
    fold :: String <- the folder of a single project
    '''
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
                    ('TRANDENS',transDen),
                    ('OVERLAP', overlap),
                    ('NAC', NAC)]

        writeH5file(outfile,outTuple)
        print('File ' + outfile + ' written.')
    else:
        log = proj + ': this calculation is not completed'
        boolean = False
    return(log,boolean)

single_inputs = namedtuple("single_input", ("glob","outF"))

def main():
    '''
    from a grid frolder to a folder ready to be propagated
    '''
    o_inputs = single_inputs("","")
    inp = read_single_arguments(o_inputs)
    folders = npArrayOfFiles(inp.glob)
    fold = folders[0:2]
    logAll = ""
    finished = 0
    problems = 0
    for i in fold:
        (log,boolean) = createOutputFile(i,inp.outF)
        logAll += '\n' + log
        if boolean:
            finished += 1
        else:
            problems += 1
    results = '''
{}

finished -> {}
problems -> {}
'''.format(logAll,finished,problems)
    print(results)


if __name__ == "__main__":
        main()



