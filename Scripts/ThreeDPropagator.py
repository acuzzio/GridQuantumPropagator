''' This script launches a 3d propagation from a h5 file. '''

import yaml
import os

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import fromFsToAu


def read_single_arguments(single_inputs):
    '''
    This funcion reads the command line arguments and assign the values on
    the namedTuple for the 3D grid propagator.
    '''
    parser = ArgumentParser()
    parser.add_argument("-o", "--output",
                        dest="o",
                        type=str,
                        help="output folder")
    parser.add_argument("-f", "--foldH5",
                        dest="f",
                        type=str,
                        help="H5 folder input")
    parser.add_argument("-i", "--input",
                        dest="i",
                        required=True,
                        type=str,
                        help="inputFile")

    args = parser.parse_args()

    if args.o != None:
        single_inputs = single_inputs._replace(out_folder=args.o)
    if args.f != None:
        single_inputs = single_inputs._replace(inputFolder=args.f)
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
            ("out_folder",
             "inputFolder",
             "inputFile"
            )
            )

defaultYaml = '''
# propagation details in fs.
dt : 0.01
states : 8
fullTime : 5

# PULSE SPECIFICATIONS
# E :: intensities
# omega ::
# sigma ::
# phi ::
# T_0 ::

pulseX : [0.0,0.0,0.0,0.0,0.0,0.0]
pulseY : [0.0,0.0,0.0,0.0,0.0,0.0]
pulseZ : [0.0,0.0,0.0,0.0,0.0,0.0]
'''

def bring_input_to_AU(iDic):
    '''
    this function is here to make the conversions between fs/ev and AU
    inputDict :: Dict
    '''
    iDic['dt'] = fromFsToAu(iDic['dt'])
    iDic['fullTime'] = fromFsToAu(iDic['fullTime'])
    return (iDic)

def main():
    '''
    This launches a 3d wavepacket propagation.
    '''
    default = single_inputs(".",              # out_folder
                            ".",              # input file
                            "."               # input h5 folder
                           )
    inputs = read_single_arguments(default)
    fn = inputs.inputFile
    if os.path.exists(fn):
        ff = loadInputYAML(fn)
        inputAU = bring_input_to_AU(ff)
        print(inputAU)
    else:
        print('File ' + fn + ' does not exist. Creating a skel one')
        with open(fn,'w') as f:
            f.write(defaultYaml)

if __name__ == "__main__":
    main()
