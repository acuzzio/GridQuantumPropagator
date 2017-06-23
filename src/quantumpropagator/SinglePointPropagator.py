'''
This module launches single point propagations...
'''

import numpy as np
import itertools as it
from collections import namedtuple

import quantumpropagator.h5Reader as hf
import quantumpropagator.Propagator as Pr
import quantumpropagator.EMPulse as pp
import quantumpropagator.GeneralFunctions as gf
import quantumpropagator.graph as gg
import quantumpropagator.commandlineParser as cmd


def printEvenergy(h5fn):
    '''
    it displays the difference in energy between states in a h5 file
    '''
    ene = hf.retrieve_hdf5_data(h5fn, 'SFS_ENERGIES')
    enezero = ene - (ene[0])
    vectorFor = "{:3.7f}"
    enezero_string = ' '.join(map(vectorFor.format, enezero))
    enezero_electronvolt_string = ' '.join(map(vectorFor.format,
        gf.HartoEv(enezero)))
    output = '\nEnergies requested:\n Hartree  {} \n Ev       {}\n'.format(
            enezero_string, enezero_electronvolt_string)
    print(output)


def single_point_propagation(h5fn, h, ts, specPulse, systemName, graph, fileO,
        outFolder):
    '''
    h5fn :: FilePath - path of h5 file
    h :: Double - time step
    ts :: Int - Number of steps
    specPulse :: function - pulse function
    systemName :: String - name of the system (for graph label)
    graph :: Bool - turn off/on graph generation
    fileO :: Bool - turn off/on file output generation
    outFolder :: FilePath - folder to save outputs

    tdp :: [3[[Double]]]
    ene :: [Double]
    enezero = the energies are taken to be ground state = 0
    mat_v = energies needs to be in a diagonal matrix
    matMu = for now, we take them like this, until we modify Molcas
    nstates = number of roots of the calculation
    states = array of population :: [Complex]
    '''

    [tdp, ene] = hf.retrieve_hdf5_data(h5fn, ['PROPERTIES', 'SFS_ENERGIES'])
    enezero = ene - (ene[0])
    mat_v = np.diag(enezero)
    matMu = tdp[0:3]
    nstates = ene.size
    states = gf.groundState(nstates)
#    print(states,nstates,matMu,mat_v,enezero,tdp,ene)

    #INITIAL VALUES
    t     = 0        # initial time
    times = []       # initial value in the times array

    np.set_printoptions(precision=14,suppress=False)

    #projectname
    longfilename = ''.join(it.takewhile(lambda x: x != '.',h5fn))
    finalName = 'Results' + longfilename + "-Ts" + str(h)
    if graph:                     # initialize repa arrays for graphics
        print("graph ON")
        statesArr = np.empty((0,nstates))
        pulseArr = np.empty((0,3))

    print('\n')

    for ii in range(ts):
        states = Pr.rk4Ene(Pr.derivativeC, t, states, h, specPulse, mat_v, matMu) # Amplitudes
        muOft = np.real(gf.dipoleMoment(states,matMu)) # Dipole moments
        norm = np.linalg.norm(states) # Norms
    #    norms     = norms+[norm]
        times = times+[t]
        t = t + h
        pulseV = pp.specificPulse(t)
        statesA = gf.abs2(states)
        tStr = '{:3.2f}'.format(t)
        vectorFor = "{:.7f}"
        statesStr = " ".join(map(vectorFor.format, statesA))
        pulseVStr = ' '.join(map(vectorFor.format, pulseV))
        muOftStr = ' '.join(map(vectorFor.format, muOft))
        normStr = '{:20.18f}'.format(1.0-norm)
        stringout = ' '.join([tStr,pulseVStr,statesStr,muOftStr,normStr])
        if fileO: # to print results in a file.dat
           ffname = finalName + ".dat"
           with open(ffname, "a") as myfile:
                myfile.write(stringout + '\n')
        else:
           print(stringout)
        if graph:
            statesArr = np.append(statesArr,[statesA],axis=0)
            pulseArr = np.append(pulseArr,[pulseV],axis=0)
            imagefn = finalName + '.png'

    if fileO:
       print('File mode ON. A new file: ' + ffname + ' has been written')

    print('\nFinal norm deviation from 1: ', '{:1.2e}'.format(1-(np.linalg.norm(states))),"\n")

    if graph:
       gg.makePulseSinglePointGraph(times, statesArr, pulseArr, imagefn,
               systemName, nstates)

single_inputs = namedtuple("single_input",
            ("out_folder",
             "nsteps",
             "dt",
             "H5file",
             "graphs",
             "outF"
            )
            )

if __name__ == "__main__":
    # single_point_propagation(h5file, timestepDT, timestepN, Graph, OutputFile)
    # h5file            String      Name of the single point h5 file
    # timestepDT        Double      dt
    # timestepN         Double      number of steps
    # pulse             [[Double]]  [t1[x,y,z],t2[x,y,z]]
    #                               extern electromagnetic field at each t(x,y,z)
    # Graph             True/False  To make the graph
    # OutputFile        True/False  To write an external file
    inputs = single_inputs("here",
                           5,
                           0.04,
                           "cisbutadieneOUT.0.rassi.h5",
                           True,
                           True
                           )
    new_inp = cmd.read_single_arguments(inputs)
    single_point_propagation(new_inp.H5file, new_inp.dt, new_inp.nsteps, pp.specificPulse,
            'CisButa', new_inp.graphs, new_inp.outF, new_inp.out_folder)
