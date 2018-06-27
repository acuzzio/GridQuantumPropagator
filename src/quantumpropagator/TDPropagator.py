''' this is the module for the hamiltonian '''

import numpy as np
import os
from quantumpropagator import (printDict, printDictKeys, loadInputYAML, bring_input_to_AU,
         warning, labTranformA, gaussian2, makeJustAnother2DgraphComplex,
         fromHartreetoCmMin1, makeJustAnother2DgraphMULTI,derivative3d,rk4Ene3d,derivative1dPhi,
         good, asyncFun, derivative1dGam, create_enumerated_folder, fromCmMin1toFs,
         makeJustAnother2DgraphComplexALLS, derivative2dGamThe, retrieve_hdf5_data,
         writeH5file, writeH5fileDict, heatMap2dWavefunction, abs2, fromHartoEv,
         makeJustAnother2DgraphComplexSINGLE, fromLabelsToFloats, derivative2dGamTheMu, pulZe,
         graphic_Pulse,derivative3dMu,equilibriumIndex)
from quantumpropagator.CPropagator import *

def expandcube(inp):
    return inp

def propagate3D(dataDict, inputDict):
    '''
    Two dictionaries, one from data file and one from input file
    it starts and run the 3d propagation of the wavefunction...
    '''
    printDict(inputDict)
    printDictKeys(dataDict)
    printDictKeys(inputDict)

    #startState = inputDict['states']
    _, _, _, nstates = dataDict['potCube'].shape
    phiL, gamL, theL, natoms, _ = dataDict['geoCUBE'].shape

    # INITIAL WF default values
    if 'factor' in inputDict:
        factor = inputDict['factor']
    else:
        factor = 1

    if 'displ' in inputDict:
        displ = inputDict['displ']
    else:
        displ = (0,0,0)

    if 'init_mom' in inputDict:
        init_mom = inputDict['init_mom']
    else:
        init_mom = (0,0,0)

    wf = np.zeros((phiL, gamL, theL, nstates), dtype=complex)
    wf[:,:,:,0] = initialCondition3d(wf[:,:,:,0],dataDict,factor,displ,init_mom)

    # Take values array from labels (radians already)
    phis,gams,thes = fromLabelsToFloats(dataDict)

    # take step
    dphi = phis[0] - phis[1]
    dgam = gams[0] - gams[1]
    dthe = thes[0] - thes[1]


    inp = { 'h'        : inputDict['dt'],
            'fullTime' : inputDict['fullTime'],
            'phiL'     : phiL,
            'gamL'     : gamL,
            'theL'     : theL,
            'natoms'   : natoms,
            'phis'     : phis,
            'gams'     : gams,
            'thes'     : thes,
            'dphi'     : dphi,
            'dgam'     : dgam,
            'dthe'     : dthe,
            'potCube'  : dataDict['potCube'],
            'kinCube'  : dataDict['kinCube'],
            'dipCube'  : dataDict['dipCUBE'],
            'pulseX'   : inputDict['pulseX'],
            'pulseY'   : inputDict['pulseY'],
            'pulseZ'   : inputDict['pulseZ'],
            'nstates'  : nstates,
            'kind'     : inputDict['kind'],
            }

    #########################################
    # Here the cube expansion/interpolation #
    #########################################

    inp = expandcube(inp)

    ########################################
    # Potentials to Zero and normalization #
    ########################################

    inp['potCube'] = dataDict['potCube'] - np.amin(dataDict['potCube'])
    norm_wf = np.linalg.norm(wf)
    good('starting NORM deviation : {}'.format(1-norm_wf))

    # magnify the potcube
    if 'enePot' in inputDict:
        enePot = inputDict['enePot']
        inp['potCube'] = inp['potCube'] * enePot
        if enePot == 0:
            warning('This simulation is done with zero Potential energy')
        elif enePot == 1:
            good('Simulation done with original Potential energy')
        else:
            warning('The potential energy has been magnified {} times'.format(enePot))

    # constant the kinCube
    kinK = False
    if kinK:
        kokoko = 1000
        inp['kinCube'] = inp['kinCube']*kokoko
        warning('kincube divided by {}'.format(kokoko))



    nameRoot = create_enumerated_folder(inputDict['outFol'])
    inputDict['outFol'] = nameRoot
    inp['outFol'] = nameRoot
    numStates = inputDict['states']


    ################
    # slice states #
    ################

    kind = inp['kind']

    # Take equilibrium points # this can be deduced by directionFile
    gsm_phi_ind, gsm_gam_ind, gsm_the_ind = equilibriumIndex(inputDict['directions1'],dataDict)

    inp['nstates'] = numStates
    if kind == '3d':
        inp['potCube'] = inp['potCube'][:,:,:,:numStates]
        inp['kinCube'] = inp['kinCube'][:,:,:]
        inp['dipCube'] = inp['dipCube'][:,:,:,:,:numStates,:numStates]
        wf =                         wf[:,:,:,:numStates]
        good('Propagation in 3D.')
        print('\nDimensions:\nPhi: {}\nGam: {}\nThet: {}\nNstates: {}\nNatoms: {}\n'.format(phiL, gamL, theL,numStates, natoms))
    elif kind == 'Phi':
        inp['potCube'] = inp['potCube'][:,gsm_gam_ind,gsm_the_ind,:numStates]
        inp['kinCube'] = inp['kinCube'][:,gsm_gam_ind,gsm_the_ind]
        inp['dipCube'] = inp['dipCube'][:,gsm_gam_ind,gsm_the_ind,:,:numStates,:numStates]
        wf             =             wf[:,gsm_gam_ind,gsm_the_ind,:numStates]
        good('Propagation in PHI with Gam {} and The {}'.format(gsm_gam_ind,gsm_the_ind))
        print('Shapes: P:{} K:{} W:{} D:{}'.format(inp['potCube'].shape, inp['kinCube'].shape, wf.shape, inp['dipCube'].shape))
        print('\nDimensions:\nPhi: {}\nNstates: {}\nNatoms: {}\n'.format(phiL, numStates, natoms))
        norm_wf = np.linalg.norm(wf)
        wf = wf / norm_wf
    elif kind == 'Gam':
        inp['potCube'] = inp['potCube'][gsm_phi_ind,:,gsm_the_ind,:numStates]
        inp['kinCube'] = inp['kinCube'][gsm_phi_ind,:,gsm_the_ind]
        inp['dipCube'] = inp['dipCube'][gsm_phi_ind,:,gsm_the_ind,:,:numStates,:numStates]
        wf             =             wf[gsm_phi_ind,:,gsm_the_ind,:numStates]
        good('Propagation in GAM with Phi {} and The {}'.format(gsm_phi_ind,gsm_the_ind))
        print('Shapes: P:{} K:{} W:{} D:{}'.format(inp['potCube'].shape, inp['kinCube'].shape, wf.shape, inp['dipCube'].shape))
        print('\nDimensions:\nGam: {}\nNstates: {}\nNatoms: {}\n'.format(gamL, numStates, natoms))
        norm_wf = np.linalg.norm(wf)
        wf = wf / norm_wf
    elif kind == 'The':
        inp['potCube'] = inp['potCube'][gsm_phi_ind,gsm_gam_ind,:,:numStates]
        inp['kinCube'] = inp['kinCube'][gsm_phi_ind,gsm_gam_ind,:]
        inp['dipCube'] = inp['dipCube'][gsm_phi_ind,gsm_gam_ind,:,:,:numStates,:numStates]
        wf             =             wf[gsm_phi_ind,gsm_gam_ind,:,:numStates]
        good('Propagation in THE with Phi {} and Gam {}'.format(gsm_phi_ind,gsm_gam_ind))
        print('Shapes: P:{} K:{} W:{} D:{}'.format(inp['potCube'].shape, inp['kinCube'].shape, wf.shape, inp['dipCube'].shape))
        print('\nDimensions:\nThe: {}\nNstates: {}\nNatoms: {}\n'.format(theL, numStates, natoms))
        norm_wf = np.linalg.norm(wf)
        wf = wf / norm_wf

    # take a wf from file (and not from initial condition)
    if 'initialFile' in inputDict:
        warning('we are taking initial wf from file')
        wffn = inputDict['initialFile']
        print('File -> {}'.format(wffn))
        wf_not_norm = retrieve_hdf5_data(wffn,'WF')
        wf = wf_not_norm/np.linalg.norm(wf_not_norm)

    #############################
    # INTEGRATOR SELECTION HERE #
    #############################

    integrators = {'3d'  : (CextractEnergy3dMu, Cderivative3dMu),
                   'Phi' : (Cenergy_1D_Phi, Cderivative_1D_Phi),
                   'Gam' : (Cenergy_1D_Gam, Cderivative_1D_Gam),
                   'The' : (Cenergy_1D_The, Cderivative_1D_The),
                  }

    CEnergy, Cpropagator = integrators[kind]

    # INITIAL DYNAMICS VALUES
    h = inp['h']
    t = 0
    counter  = 0
    fulltime = inp['fullTime']
    fulltimeSteps = int(fulltime/h)
    deltasGraph = inputDict['deltasGraph']
    print('I will do {} steps.\n'.format(fulltimeSteps))
    outputFile = os.path.join(nameRoot, 'output')
    outputFileP = os.path.join(nameRoot, 'outputPopul')
    print('\ntail -f {}\n'.format(outputFileP))

    # saving input data in h5 file
    dataH5filename = os.path.join(nameRoot, 'allInput.h5')
    writeH5fileDict(dataH5filename,inp)

    # print top of table
    header = ' Coun |  step N   |       fs   |  NORM devia.  | Kin. Energy  | Pot. Energy  | Total Energy | Tot devia.   |    Pulse X    |    Pulse Y    |    Pulse Z    |'
    bar = ('-' * (len(header)))
    print('Energies in ElectronVolt \n{}\n{}\n{}'.format(bar,header,bar))

    # calculating initial total/potential/kinetic
    kin, pot = CEnergy(t,wf,inp)
    kinetic = np.vdot(wf,kin)
    potential = np.vdot(wf,pot)
    initialTotal = kinetic + potential
    inp['initialTotal'] = initialTotal.real

    # to give the graph a nice range
    inp['vmax_value'] = abs2(wf).max()

    # graph the pulse
    graphic_Pulse(inp)

    for ii in range(fulltimeSteps):
        if (ii % deltasGraph) == 0 or ii==fulltimeSteps-1:
            #  async is awesome
            doAsyncStuffs(wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy)
            #asyncFun(doAsyncStuffs,wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy)
            counter += 1

        wf = Crk4Ene3d(Cpropagator,t,wf,inp)
        t  = t + h


def doAsyncStuffs(wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy):
    nameRoot = inputDict['outFol']
    nstates = inp['nstates']
    name = os.path.join(nameRoot, 'Gaussian' + '{:04}'.format(counter))
    h5name = name + ".h5"
    writeH5file(h5name,[("WF", wf),("Time", [t/41.5,t])])
    kin, pot = CEnergy(t,wf,inp)

    #kinetic = np.vdot(wf,kin)
    #potential = np.vdot(wf,pot)

    kinetic = np.vdot(wf,kin)
    potential = np.vdot(wf,pot)
    total = kinetic + potential
    initialTotal = inp['initialTotal']
    norm_wf = np.linalg.norm(wf)

    ##  you wanted to print the header when the table goes off screen... this is why you get the rows number
    #rows, _ = os.popen('stty size', 'r').read().split()
    #if int(rows) // counter == 0:
    #    print('zero')

    outputStringS = ' {:04d} |{:10d} |{:11.4f} | {:+e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+13.5e} | {:+13.5e} | {:+13.5e} |'
    outputString = outputStringS.format(counter, ii,t/41.3,1-norm_wf,fromHartoEv(kinetic.real),fromHartoEv(potential.real),fromHartoEv(total.real),fromHartoEv(initialTotal - total.real), pulZe(t,inp['pulseX']), pulZe(t,inp['pulseY']), pulZe(t,inp['pulseZ']) )
    print(outputString)

    kind = inp['kind']
    outputStringSP = "{:11.4f}".format(t/41.3)
    for i in range(nstates):
        if kind == '3d':
            singleStatewf = wf[:,:,:,i]
        elif kind == 'Phi' or kind == 'Gam' or kind == 'The':
            singleStatewf = wf[:,i]
        outputStringSP += " {:+7.5e} ".format(np.linalg.norm(singleStatewf)**2)


    with open(outputFileP, "a") as oofP:
        oofP.write(outputStringSP + '\n')

    with open(outputFile, "a") as oof:
        outputStringS2 = '{} {} {} {} {} {} {} {} {} {} {}'
        outputString2 = outputStringS2.format(counter,ii,t/41.3,1-norm_wf,fromHartoEv(kinetic.real),fromHartoEv(potential.real),fromHartoEv(total.real),fromHartoEv(initialTotal - total.real), pulZe(t,inp['pulseX']), pulZe(t,inp['pulseY']), pulZe(t,inp['pulseZ']))
        oof.write(outputString2 + '\n')

    #####################
    # on the fly graphs #
    #####################

    if 'graphs' in inputDict:
        vmaxV = inp['vmax_value']
        oneD = True
        if oneD:
            graphFileName = name + ".png"
            if kind == 'Phi':
                valuesX = inp['phis']
                label = 'Phi {:11.4f}'.format(t/41.3)
            elif kind == 'Gam':
                valuesX = inp['gams']
                label = 'Gam {:11.4f}'.format(t/41.3)
                pot=inp['potCube'][0]
            elif kind == 'The':
                valuesX = inp['thes']
                label = 'The {:11.4f}'.format(t/41.3)
            makeJustAnother2DgraphComplexSINGLE(valuesX,wf,graphFileName,label)
        twoD = False
        if twoD:
            for i in range(nstates):
                for j in range(i+1): # In python the handshakes are like this...
                    graphFileName = '{}_state_{}_{}.png'.format(name,i,j)
                    heatMap2dWavefunction(wf[:,:,i],wf[:,:,j],graphFileName,t/41.3,vmaxV)

def forcehere(vec,ind,h=None):
    '''
    calculates the numerical force at point at index index
    vector :: np.array(double)
    index :: Int
    '''
    if h == None:
        warning('dimensionality is not clear')
        h = 1
    num = (-vec[ind-2]+16*vec[ind-1]-30*vec[ind]+16*vec[ind+1]-vec[ind+2])
    denom = 12 * h**2
    return(num/denom)


def initialCondition3d(wf, dataDict, factor=None, displ=None, init_mom=None):
    '''
    calculates the initial condition WV
    wf :: np.array(phiL,gamL,theL)  Complex
    datadict :: Dictionary {}
    '''
    good('Initial condition printing')

    # Take equilibrium points
    gsm_phi_ind = dataDict['phis'].index('P000-000')
    gsm_gam_ind = dataDict['gams'].index('P016-923')
    gsm_the_ind = dataDict['thes'].index('P114-804')

    # Take values array from labels
    phis,gams,thes = fromLabelsToFloats(dataDict)

    # take step
    dphi = phis[0] - phis[1]
    dgam = gams[0] - gams[1]
    dthe = thes[0] - thes[1]

    # take range
    range_phi = phis[-1] - phis[0]
    range_gam = gams[-1] - gams[0]
    range_the = thes[-1] - thes[0]

    # slice out the parabolas at equilibrium geometry
    pot = dataDict['potCube']
    parabola_phi = pot[:,gsm_gam_ind,gsm_the_ind,0]
    parabola_gam = pot[gsm_phi_ind,:,gsm_the_ind,0]
    parabola_the = pot[gsm_phi_ind,gsm_gam_ind,:,0]

    # calculate force with finite difference  # WATCH OUT RADIANS AND ANGLES HERE 
    force_phi = forcehere(parabola_phi, gsm_phi_ind, h=dphi)
    force_gam = forcehere(parabola_gam, gsm_gam_ind, h=dgam)
    force_the = forcehere(parabola_the, gsm_the_ind, h=dthe)

    # Now, I want the coefficients of the second derivative of the kinetic energy jacobian
    # for the equilibrium geometry, so that I can calculate the gaussian.
    # in the diagonal approximation those are the diagonal elements, thus element 0,4,8.

    coe_phi = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,0,2]
    #coe_gam = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,4,2]
    coe_gam = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,0,2]
    warning('coe_gam has been changed !!! in initialcondition function')
    coe_the = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,8,2]

    # they need to be multiplied by (-2 * hbar**2), where hbar is 1. And inverted, because the MASS
    # is at denominator, and we kind of want the mass...
    G_phi = 1 / ( -2 * coe_phi )
    G_gam = 1 / ( -2 * coe_gam )
    G_the = 1 / ( -2 * coe_the )

    # factor is just to wide the gaussian a little bit leave it to one.
    factor = factor or 1
    if factor != 1:
        warning('You have a factor of {} enabled on initial condition'.format(factor))
    G_phi = G_phi/factor
    G_gam = G_gam/factor
    G_the = G_the/factor

    Gw_phi = np.sqrt(force_phi*G_phi)
    Gw_gam = np.sqrt(force_gam*G_gam)
    Gw_the = np.sqrt(force_the*G_the)

    w_phi = np.sqrt(force_phi/G_phi)
    w_gam = np.sqrt(force_gam/G_gam)
    w_the = np.sqrt(force_the/G_the)

    # displacements from equilibrium geometry
    displ = displ or (0,0,0)
    displPhi,displGam,displThe = displ
    if displPhi != 0 or displGam != 0 or displThe != 0:
        warning('Some displacements activated | Phi {} | Gam {} | The {}'.format(displPhi,displGam,displThe))

    phi0 = phis[gsm_phi_ind + displPhi]
    gam0 = gams[gsm_gam_ind + displGam]
    the0 = thes[gsm_the_ind + displThe]

    # initial moments?
    init_mom = init_mom or (0,0,0)
    init_momPhi,init_momGam,init_momThe = init_mom
    if init_momPhi != 0 or init_momGam != 0 or init_momThe != 0:
        warning('Some inititial moment is activated | Phi {} | Gam {} | The {}'.format(init_momPhi,init_momGam,init_momThe))

    for p, phi in enumerate(phis):
        phiV = gaussian2(phi, phi0, Gw_phi, init_momPhi)
        for g, gam in enumerate(gams):
            gamV = gaussian2(gam, gam0, Gw_gam, init_momGam)
            for t , the in enumerate(thes):
                theV = gaussian2(the, the0, Gw_the, init_momThe)
                #print('I: {}\tV: {}\tZ: {}\tG: {}\t'.format(t,the,the0,theV))

                wf[p,g,t] = phiV * gamV * theV

    norm_wf = np.linalg.norm(wf)
    print('NORM before normalization: {:e}'.format(norm_wf))
    print('Steps: phi({:.3f}) gam({:.3f}) the({:.3f})'.format(dphi,dgam,dthe))
    print('Range: phi({:.3f}) gam({:.3f}) the({:.3f})'.format(range_phi,range_gam,range_the))
    wf = wf / norm_wf
    print(wf.shape)
    print('\n\nparabola force constant: {:e} {:e} {:e}'.format(force_phi,force_gam,force_the))
    print('values on Jacobian 2nd derivative: {:e} {:e} {:e}'.format(coe_phi,coe_gam,coe_the))
    print('G: {:e} {:e} {:e}'.format(G_phi,G_gam,G_the))
    print('Gw: {:e} {:e} {:e}'.format(Gw_phi,Gw_gam,Gw_the))
    print('w: {:e} {:e} {:e}'.format(w_phi,w_gam,w_the))
    print('cm-1: {:e} {:e} {:e}'.format(fromHartreetoCmMin1(w_phi),
                                  fromHartreetoCmMin1(w_gam),
                                  fromHartreetoCmMin1(w_the)))
    print('fs: {:e} {:e} {:e}'.format(fromCmMin1toFs(w_phi),
                                fromCmMin1toFs(w_gam),
                                fromCmMin1toFs(w_the)))

    return(wf)


if __name__ == "__main__":
    fn1 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/input.yml'
    fn2 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/datainput.npy'
    inputDict = bring_input_to_AU(loadInputYAML(fn1))
    dataDict = np.load(fn2) # this is a numpy wrapper, for this we use [()]
    propagate3D(dataDict[()], inputDict)



    ## REDUCE THE PROBLEM IN 1d GAM 1 state 
    ## Take equilibrium points
    #gsm_phi_ind = dataDict['phis'].index('P000-000')
    #gsm_gam_ind = dataDict['gams'].index('P016-923')
    #gsm_the_ind = dataDict['thes'].index('P114-804')

    ##print(inp['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind])

    ## slice the grid
    #inp['potCube'] = inp['potCube'][gsm_phi_ind,:,gsm_the_ind,0]
    #inp['kinCube'] = inp['kinCube'][gsm_phi_ind,:,gsm_the_ind]
    #wf =                         wf[gsm_phi_ind,:,gsm_the_ind]


    ## REDUCE THE PROBLEM IN 2d THE GAM 1 state
    ## Take equilibrium points
    #gsm_phi_ind = dataDict['phis'].index('P000-000')
    #gsm_gam_ind = dataDict['gams'].index('P016-923')
    #gsm_the_ind = dataDict['thes'].index('P114-804')

    ## slice the grid
    #inp['potCube'] = inp['potCube'][gsm_phi_ind,:,:,:numStates]
    #inp['kinCube'] = inp['kinCube'][gsm_phi_ind,:,:]
    #inp['dipCube'] = inp['dipCube'][gsm_phi_ind,:,:,:,:numStates,:numStates]
    #wf =                         wf[gsm_phi_ind,:,:,:numStates]
    #wf = wf/np.linalg.norm(wf)
    #str111 = 'potCube: {}\nkinCube: {}\ndipCube: {}\nwf:      {}'
    #print(str111.format(inp['potCube'].shape,inp['kinCube'].shape,inp['dipCube'].shape,wf.shape))
