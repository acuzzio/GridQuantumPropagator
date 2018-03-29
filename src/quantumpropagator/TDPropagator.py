''' this is the module for the hamiltonian '''

import numpy as np
from quantumpropagator import (printDict, printDictKeys, loadInputYAML, bring_input_to_AU)

def propagate3D(dataDict, inputDict):
    '''
    Two dictionaries, one from data file and one from input file
    it starts and run the 3d propagation of the wavefunction...
    '''
    printDictKeys(dataDict)
    printDict(inputDict)
    printDictKeys(inputDict)

    h = inputDict['dt']
    startState = inputDict['states']



if __name__ == "__main__":
    fn1 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/hugo.yml'
    fn2 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/datahugo.npy'
    inputDict = bring_input_to_AU(loadInputYAML(fn1))
    dataDict = np.load(fn2) # this is a numpy wrapper, for this we use [()]
    propagate3D(dataDict[()], inputDict)
