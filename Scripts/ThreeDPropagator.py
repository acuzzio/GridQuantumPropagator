''' This script launches a 3d propagation from a h5 file. '''

import glob
import numpy as np
import os

from argparse import ArgumentParser
from collections import namedtuple
from quantumpropagator import (fromFsToAu,
        ensure_dir,printDict,stringTransformation3d,retrieve_hdf5_data,
        loadInputYAML, readDirectionFile, err, good, propagate3D, bring_input_to_AU,
        printProgressBar)


def read_single_arguments(single_inputs):
    '''
    This funcion reads the command line arguments and assign the values on
    the namedTuple for the 3D grid propagator.
    '''
    parser = ArgumentParser()
    parser.add_argument("-i", "--input",
                        dest="i",
                        required=True,
                        type=str,
                        help="inputFile")

    args = parser.parse_args()

    if args.i != None:
        single_inputs = single_inputs._replace(inputFile=args.i)
    return single_inputs

single_inputs = namedtuple("single_input",
            ("inputFile"
            )
            )

defaultYaml = '''# folders
proj_label : zNorbornadiene_
inputFol : /home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/b-corrected
outFol : /home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/results

# propagation details in fs.
dt : 0.01
states : 8
fullTime : 5

# PULSE SPECIFICATIONS
# E :: intensities
# omega ::
# sigma :: length in fs
# phi ::
# T_0 :: center of the packet in fs

pulseX : [0.0,0.0,0.0,0.0,0.0]
pulseY : [0.0,0.0,0.0,0.0,0.0]
pulseZ : [0.0,0.0,0.0,0.0,0.0]

directions1 : /home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/directions1
directions2 : /home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/directions2

'''

def revListString(a):
    '''
    Just need to order my strings when the negative is there...
    '''
    return([x for x in a if x[0]=='N'][::-1] + [x for x in a if x[0]=='P'])

def readDirections(dir1,dir2):
    '''
    The propagation will happen in the cube that is generated by reading two
    directions file. They are the same format as the ones used in bash to run jobs
    '''
    phi1, gam1, the1 = readDirectionFile(dir1)
    phi2, gam2, the2 = readDirectionFile(dir2)
    # want the right format here, it does not matter the order anymore like in the
    # correction step !!
    phis = phi2[::-1] + phi1[1:]
    gams = gam2[::-1] + gam1[1:]
    thes = the2[::-1] + the1[1:]
    return(phis, gams, thes)

def main():
    '''
    This will launch a 3d wavepacket propagation.
    '''
    default = single_inputs(".")              # input file
    inputs = read_single_arguments(default)
    fn = inputs.inputFile
    if os.path.exists(fn):
        ff = loadInputYAML(fn)
        inputAU = bring_input_to_AU(ff)
        # create subfolder with yml file name
        filename, file_extension = os.path.splitext(fn)
        projfolder = os.path.join(inputAU['outFol'],filename)
        ensure_dir(projfolder)
        inputAU['outFol'] = projfolder
        print('\nNEW 3D PROPAGATION')
        # is there a data file?
        if 'dataFile' in inputAU:
            data = np.load(inputAU['dataFile'])

            # LAUNCH THE PROPAGATION, BITCH 
            # [()] <- because np.load returns a numpy wrapper on the dictionary

            propagate3D(data[()], inputAU)

        else:
            # if not, guess you should create it...
            good('data file creation in progress...')
            phis, gams, thes = readDirections(inputAU['directions1'],inputAU['directions2'])

            # read the first one to understand who is the seed of the cube and take numbers
            phi1, gam1, the1 = readDirectionFile(inputAU['directions1'])
            ext = '.corrected.h5'
            prjlab = inputAU['proj_label']
            first_file = inputAU['proj_label'] + phi1[0] + '_' + gam1[0] + '_' + the1[0] + ext
            fnh5 = os.path.join(inputAU['inputFol'], first_file)
            nstates = len(retrieve_hdf5_data(fnh5,'ROOT_ENERGIES'))
            natoms = len(retrieve_hdf5_data(fnh5,'CENTER_COORDINATES'))
            lengths = '\nnstates: {}\nnatoms:  {}\nphi:     {}\ngamma:   {}\ntheta:   {}'
            phiL, gamL, theL = len(phis), len(gams), len(thes)
            output = lengths.format(nstates, natoms, phiL, gamL, theL)

            # start to allocate the vectors
            potCUBE = np.empty((phiL, gamL, theL, nstates))
            kinCUBE = np.empty((phiL, gamL, theL, 9, 3))
            dipCUBE = np.empty((phiL, gamL, theL, 3, nstates, nstates))
            geoCUBE = np.empty((phiL, gamL, theL, natoms, 3))

            for p, phi in enumerate(phis):
                for g, gam in enumerate(gams):
                    for t, the in enumerate(thes):
                        labelZ = prjlab + phi + '_' + gam + '_' + the + ext
                        fnh5 = os.path.join(inputAU['inputFol'], labelZ)
                        if os.path.exists(fnh5):
                            potCUBE[p,g,t] = retrieve_hdf5_data(fnh5,'ROOT_ENERGIES')
                            kinCUBE[p,g,t] = retrieve_hdf5_data(fnh5,'KINETIC_COEFFICIENTS')
                            dipCUBE[p,g,t] = retrieve_hdf5_data(fnh5,'DIPOLES')
                            geoCUBE[p,g,t] = retrieve_hdf5_data(fnh5,'CENTER_COORDINATES')
                        else:
                            err('{} does not exist'.format(labelZ))
                    printProgressBar(p*gamL+g,phiL*gamL,prefix = 'H5 data loaded:')

            data = {'kinCube' : kinCUBE,
                    'potCube' : potCUBE,
                    'dipCUBE' : dipCUBE,
                    'geoCUBE' : geoCUBE,
                    'phis'    : phis,
                    'gams'    : gams,
                    'thes'    : thes
                    }
            np.save('data' + filename, data)
            with open(fn, 'a') as f:
                stringAdd = 'dataFile : data' + filename + '.npy'
                f.write(stringAdd)
            print('\n...done!\n')

    else:
        filename, file_extension = os.path.splitext(fn)
        if file_extension == '':
            fn = fn + '.yml'
        good('File ' + fn + ' does not exist. Creating a skel one')
        with open(fn,'w') as f:
            f.write(defaultYaml)

if __name__ == "__main__":
    main()
