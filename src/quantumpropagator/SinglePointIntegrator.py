
import numpy as np
import itertools as it

import quantumpropagator.h5Reader as hf
import quantumpropagator.Propagator as Pr
import quantumpropagator.pulse as pp
import quantumpropagator.GeneralFunctions as gf
import quantumpropagator.graph as gg



def singlePointIntegration(h5fn, h, ts, specPulse, graph, systemName, fileO):
    '''
    Molcas parsers
    h  :: Double    time step
    ts :: Int       Number of steps
    fn :: FilePath
    tdp :: [3[[Double]]]
    ene :: [Double]

    eneZero = the energies are taken to be ground state = 0
    matV = energies needs to be in a diagonal matrix
    matMu = for now, we take them like this, until we modify Molcas
    nstates = number of roots of the calculation
    states = array of population :: [Complex]
    '''

    [tdp,ene] = hf.retrieve_hdf5_data(fn,['MLTPL','SFS_ENERGIES'])
    eneZero = ene - (ene[0])
    matV = np.diag(eneZero)
    matMu = tdp[0:3]
    nstates = ene.size
    states = gf.groundState(nstates)

    #INITIAL VALUES
    t     = 0        # initial time
    times = []       # initial value in the times array

    np.set_printoptions(precision=14,suppress=False)

    #projectname
    longfilename = ''.join(it.takewhile(lambda x: x != '.',fn))
    finalName = 'Results' + longfilename + "-Ts" + str(h)

    if graph:                     # initialize repa arrays for graphics
       print("graph ON")
       statesArr = np.empty((0,nstates))
       pulseArr = np.empty((0,3))

    print('\n')

    for ii in range(ts):
        states = Pr.rk4Ene(Pr.derivativeC, t, states, h, specPulse, matV, matMu) # Amplitudes
        muOft = np.real(gf.dipoleMoment(states,matMu)) # Dipole moments
        norm = np.linalg.norm(states) # Norms
    #    norms     = norms+[norm]
        times = times+[t]
        t = t + h
        pulseV = pp.specificPulse(t)
        statesA = np.absolute(states)
        tStr = '{:3.2f}'.format(t)
        vectorFor = "{:.7f}"
        statesStr = " ".join(map(vectorFor.format, statesA))
        pulseVStr = ' '.join(map(vectorFor.format, pulseV))
        muOftStr = ' '.join(map(vectorFor.format, muOft))
        normStr = '{:20.18f}'.format(1.0-norm)
        stringout = ' '.join([tStr,pulseVStr,statesStr,muOftStr,normStr])
        if fileO:   # to print results in a file.dat
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


if __name__ == "__main__":
    # singlePointIntegration(h5file, timestepDT, timestepN, Graph, OutputFile)
    # h5file            String      Name of the single point h5 file
    # timestepDT        Double      dt
    # timestepN         Double      number of steps
    # pulse             [[Double]]  [t1[x,y,z],t2[x,y,z]] extern electromagnetic field at each t(x,y,z)
    # Graph             True/False  To make the graph
    # OutputFile        True/False  To write an external file

    singlePointIntegration('LiH.rassi.h5', 0.04, 10, pp.specificPulse,
            'LiHAst', True, True)

