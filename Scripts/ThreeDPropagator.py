''' This script launches a 3d propagation from a h5 file. '''

import glob
import numpy as np
import os
import yaml

from argparse import ArgumentParser
from collections import namedtuple
from quantumpropagator import (fromFsToAu,
        ensure_dir,printDict,stringTransformation3d,retrieve_hdf5_data)


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

def loadInputYAML(fn):
    '''
    this function reads the input file and returns a dictionary with inputs
    fn :: filePath
    '''
    with open(fn, 'r') as f:
         diction = yaml.load(f)
    return diction

single_inputs = namedtuple("single_input",
            ("inputFile"
            )
            )

defaultYaml = '''# folders
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
'''

def bring_input_to_AU(iDic):
    '''
    this function is here to make the conversions between fs/ev and AU
    inputDict :: Dict
    '''
    iDic['dt'] = fromFsToAu(iDic['dt'])
    iDic['fullTime'] = fromFsToAu(iDic['fullTime'])
    # change sigmas and T_0s
    iDic['pulseX'][2] = fromFsToAu(iDic['pulseX'][2])
    iDic['pulseX'][4] = fromFsToAu(iDic['pulseX'][4])
    iDic['pulseY'][2] = fromFsToAu(iDic['pulseY'][2])
    iDic['pulseY'][4] = fromFsToAu(iDic['pulseY'][4])
    iDic['pulseZ'][2] = fromFsToAu(iDic['pulseZ'][2])
    iDic['pulseZ'][4] = fromFsToAu(iDic['pulseZ'][4])
    return (iDic)

def revListString(a):
    '''
    Just need to order my strings when the negative is there...
    '''
    return([x for x in a if x[0]=='N'][::-1] + [x for x in a if x[0]=='P'])

def main():
    '''
    This launches a 3d wavepacket propagation.
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
        inputAU['outFol']=projfolder
        print('\nNEW 3D PROPAGATION')
        printDict(inputAU)

        # load h5 files
        allH5 = sorted(glob.glob(inputAU['inputFol'] + '/*.h5'))
        dime = len(allH5)
        allH5First = allH5[0]
        nstates = len(retrieve_hdf5_data(allH5First,'ROOT_ENERGIES'))
        natoms = len(retrieve_hdf5_data(allH5First,'CENTER_COORDINATES'))
        newdime = (dime * 2)-1
        bigArrayLab1 =  np.empty((dime), dtype=object)
        bigArrayLab2 =  np.empty((dime), dtype=object)
        bigArrayLab3 =  np.empty((dime), dtype=object)
        ind = 0
        for fileN in allH5:
            (axis1,str1,axis2,str2,axis3,str3) = stringTransformation3d(fileN)
            bigArrayLab1[ind] = str1
            bigArrayLab2[ind] = str2
            bigArrayLab3[ind] = str3
            ind += 1
        labelsAxis1 = revListString(np.unique(bigArrayLab1))
        labelsAxis2 = revListString(np.unique(bigArrayLab2))
        labelsAxis3 = revListString(np.unique(bigArrayLab3))
        print(labelsAxis1,labelsAxis2,labelsAxis3)


    else:
        print('File ' + fn + ' does not exist. Creating a skel one')
        with open(fn,'w') as f:
            f.write(defaultYaml)

if __name__ == "__main__":
    main()
