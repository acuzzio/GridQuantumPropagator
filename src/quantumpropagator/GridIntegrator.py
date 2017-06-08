import numpy as np
import multiprocessing
import itertools as it
import os
from collections import namedtuple
from math import sqrt

import graph as gg
import Propagator as Pr
import pulse as pp
import GeneralFunctions as gf
import GeneralFunctionsSystem as gfs
import h5Reader as h5
import initialConditions as ic
import commandlineParser as cmd
from astridParser import astridParser
from errors import err, good


def grid1DIntegrationAstrid(inputs):
    '''
    This function is the Grid1d done from ASTRID data:

    dist :: [Double]           -> gridN
    ene  :: [[Double]]         -> gridN, nstates
    dipo :: [[[[Double]]]]     -> gridN, 1, nstates, nstates
    GRID :: [[Complex]] -> nstates, gridN
    '''

    ''' prepare folder '''
    folderN     = inputs.OutFolder
    LAB         = inputs.label
    folderN     = folderN + LAB
    gfs.ensure_dir(folderN)
    nameRoot = folderN + '/' + LAB

    ''' GET and prepare INPUTS '''
    startState = 4
    startgridN = 971
    (distSmall,eneSmall,dipoSmall,NACsmall,Gelesmall) = astridParser(startState,startgridN)
    (gridNSmall,nstates) = eneSmall.shape
    h           = inputs.timeStep
    minGround   = np.min(eneSmall)            # minimium on the ground state
    eneZero     = eneSmall - minGround
    energyHar   = gf.EvtoHar(eneZero)
    deltaX      = distSmall[1] - distSmall[0]

    ''' Initial Gaussian '''
    reducedMass = ic.reducedMassLiH()
    LiHomega    = gf.fromCmMin1toHartree(1285)
    mu          = 3               # shift of the gaussian in the dist coordinate (in bohr)
    moment      = 0               # initial moment

    ''' Pulse '''
    Ed    = 0.032
    omega = 7 * 0.0367493   # to pass from Ev to Hartree 4 will excite second state in astrid data
    sigmP = 40
    phi   = 0
    t0P   = 400
    filePulse = nameRoot + 'FigurePulse.png'
    if inputs.fullTime*inputs.timeStep < 400:
        whole = 400/inputs.timeStep
    else:
        whole = inputs.fullTime
    gg.grapPulse(whole, h, Ed, omega, sigmP, phi, t0P, filePulse)

    ''' Add well or not?'''
    if inputs.expand:
        points = 60
        absorbRight = 60
        dipo = ic.expandDipo2(dipoSmall,points)
        NAC = ic.expandNAC2(NACsmall,points)
        Gele = ic.expandNAC2(Gelesmall,points)
        (dist,ene) = ic.extrapolateSxLinear(distSmall,energyHar,points)
        absorbPot = ic.absorbingPotential3Right(dist,absorbRight)
        #absorbPot = absorbPot * 0
        #dist = ic.expandDist2(distSmall,gridNSmall,deltaX,points)
        #ene  = ic.expandEne2(energyHar,points)     # Astrid Potential
        #ene  = ic.expandEneZero(energyHar) # Zero Potential
        #(ene, eneSmall)  = ic.expandEneArmonic(energyHar,reducedMass*(LiHomega**2),mu,distSmall,30) # Harmonic Potential
    else:
        dist           = distSmall
        dipo           = dipoSmall
        ene            = energyHar
        NAC            = NACsmall
        Gele           = Gelesmall

    ''' Prepare variables '''
    gridN       = dist.size
    sigma       = np.sqrt(1/(reducedMass*LiHomega))
    GRID        = ic.createInitialState(nstates, gridN, dist, mu, sigma, moment)

    ''' INITIAL VALUES '''
    t        = 0                  # initial time
    kaxisR   = Pr.createXaxisReciprocalspace1d(gridN,deltaX)
    deltaK   = kaxisR[1]-kaxisR[0]
    lengthK  = kaxisR.max()*2
    #misc     = passage(nstates,gridN,kaxisR,reducedMass)
    counter  = 0                  # to label images of frames
    counterP = '{:04}'.format(counter)
    name     = nameRoot + 'Gaussian' + counterP
    gf.asyncFun(gg.makeJustAnother2DgraphComplexALLS, dist, GRID, name,"gauss Initial")
    h5name = nameRoot + 'WaveFun' + '{:04}'.format(counter) + '.h5'
    h5.writeH5file(h5name,[("WF", GRID)])

    ''' GRAPHICS initial conditions Asyncs '''
    if inputs.dipoleMatrixGraph:
        gf.asyncFun(gg.dipoleMatrixGraph1d,inputs,name,['Z'],nstates,gridN,distSmall,dipoSmall)
    if inputs.EneGraph:
        name2         = nameRoot + 'EnergiesvsGrid.png'
        name3         = nameRoot + 'SmallEnergiesvsGrid.png'
        gf.asyncFun(gg.make2DgraphTranspose,distSmall,eneSmall,name2,"LiHAst",nstates)
        gf.asyncFun(gg.make2DgraphTranspose,dist,ene,name3,"LiHAst",nstates)


    if np.linalg.norm(absorbPot) == 0.0:
        absPstring = "NO"
    else:
        absPstring = "YES"

    pulseString    = 'Pulse | Ed={:3.6f} omega={:3.6f} sigma={:3.6f} phi={:3.6f} t0={:3.6f}'.format(Ed,omega,sigmP,phi,t0P)
    initcondString = 'Init Wavepacket | deltaX={:3.6f} reducedMass={:3.6f} LiHomega={:3.6f} sigma={:3.6f} mu={:3.6f} initial moment={:3.6f} deltaK={:3.3f} LengthK={:3.3f} nstates/Gridp=({},{}[{}]) absP = {}'.format(deltaX,reducedMass,LiHomega,sigma,mu,moment,deltaK,lengthK,nstates,gridN,startgridN,absPstring)
    initialString  = 't:    0.00000    NORM: {:12.5e}  <- initial state'.format(1-np.linalg.norm(GRID))
    outputFile     = nameRoot + 'output'
    print(pulseString + '\n' + initcondString + '\n' + initialString)
    with open(outputFile, "w") as oof:
        oof.write(str(inputs) + '\n')
        oof.write(pulseString + '\n')
        oof.write(initcondString + '\n')
        oof.write(initialString + '\n')



    ''' INTEGRATION '''
    for ii in range(inputs.fullTime):
      tStr   = '{:3.2f}'.format(t)
      pulseV = [inputs.specPulse(t,Ed,omega,sigmP,phi,t0P),inputs.specPulse(t+0.5*h,Ed,omega,sigmP,phi,t0P),inputs.specPulse(t+h,Ed,omega,sigmP,phi,t0P)] # FIX THIS AS SOON AS YOU HAVE TIME
      GRID   = Pr.rk4Ene1dSLOW(Pr.derivative1d,t,GRID,h,pulseV,ene,dipo,NAC,Gele,nstates,gridN,kaxisR,reducedMass,absorbPot)
      t     = t + h


      ''' OUTPUT '''
      if (ii % inputs.deltasGraph) == 0:
         Total  = Pr.calculateTotal(t,GRID,pulseV,ene,dipo,NAC,Gele,nstates,gridN,kaxisR,reducedMass,absorbPot)

         ''' modsquare every elements -> sum them up -> square root '''
         (pop,sumPop)  = gf.population(GRID)
         popuS = gf.ndprint(pop, format_string ='{:17.15f}')
         totalString = '{:17.10e}'.format(Total.real)
         counter += 1
         normDeviation = 1-np.linalg.norm(GRID)
         outputString = 't: {:10.5f} {:10.5f} E: {:15.5e} NORM: {:12.5e} <Et>: {} Pop: {} {:17.15f} iter: {:6} Count: {:04}'.format(
                            t/41.5,    t,      pulseV[0][2],normDeviation,totalString,popuS,sumPop,ii,      counter)
         print(outputString)
         with open(outputFile, "a") as oof:
             oof.write(outputString + '\n')

         if inputs.wfGraph:
             name = nameRoot + 'Gaussian' + '{:04}'.format(counter)
             gf.asyncFun(gg.makeJustAnother2DgraphComplexALLS, dist, np.asarray(GRID), name,"gauss " + '{:8.5f}'.format(t/41.5))
         # save wavefunction in hdf5 format
         h5name = nameRoot + 'WaveFun' + '{:04}'.format(counter) + '.h5'
         h5.writeH5file(h5name,[("WF", GRID),("Time", [t/41.5,t]),("Pulse",pulseV[0][2]),("Norm Deviation",normDeviation),("Populations",pop),("iter", ii)])


    ''' These lines added to quickly profile:
        On a first run the code creates a file with some results.
        If this file exists, every next run in the same folder will
        be matched against this, creating a "Test passed/failed" message
    '''
    filenameProf = nameRoot + 'resultsProfile'
    if os.path.exists(filenameProf):
       azer = gf.loadComplex(filenameProf)
       #if all(np.isclose(azer,GRID[0])):
       if np.array_equal(azer,GRID[0]):
           good('Test passed')
       else:
           dotP = np.vdot(azer,GRID[0])
           err('Test failed ' + '{:3.5f}'.format(dotP))
    else:
       save = (np.asarray(GRID[0]))
       gf.saveComplex(filenameProf, save)
       print('First time you run this. File saved: ', filenameProf)


##############################################################
''' NAMED TUPLES '''
inputData = namedtuple("inputData",
                            (#"glob",
                            # "nstates",
                            # "matrixDIM",
                             "OutFolder",
                             "label",
                             "expand",
                             "EneGraph",
                             "deltasGraph",
                             "singleDipGrapf",
                             "dipoleMatrixGraph",
                             "wfGraph",
                             "timeStep",
                             "fullTime",
                             "specPulse"
                             )
                      )
#passage = namedtuple("passage",(
#                          "nstates",
#                          "gridN",
#                          "kaxisR",
#                          "reducedMass")
#                    )
###############################################################

if __name__ == "__main__":
    inputs = inputData(
          #   "",                        # global expression
          #   0,                         # Number of states (this can be removed)
          #   0,                         # Number of dimension of Molcas Matrix (can be removed)
             "output-graphics/",         # output folder 
             "WorkOnNac",                # Run Label
             True,                       # Expand and create well
             True,                       # Energies Graphic
             50,                         # Graphs every N values
             False,                      # Single element Graphs of dipole
             True,                       # Matrix graphics of dipole elements
             True,                       # Wavefunction Graphic
             0.04,                       # TimeStep
             40,                         # Number of steps
             pp.varPulseZ                # External field Pulse
             )
    newInputs = cmd.readArguments(inputs)
    print(newInputs)
    grid1DIntegrationAstrid(newInputs)

