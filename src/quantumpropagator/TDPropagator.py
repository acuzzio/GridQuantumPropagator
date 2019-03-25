''' this is the module for the hamiltonian '''

import numpy as np
import os
from quantumpropagator import (printDict, printDictKeys, loadInputYAML, bring_input_to_AU,
         warning, labTranformA, gaussian2, makeJustAnother2DgraphComplex,
         fromHartreetoCmMin1, makeJustAnother2DgraphMULTI,derivative3d,rk4Ene3d,derivative1dPhi,
         good, asyncFun, derivative1dGam, create_enumerated_folder, fromCmMin1toFs,
         makeJustAnother2DgraphComplexALLS, derivative2dGamThe, retrieve_hdf5_data,
         writeH5file, writeH5fileDict, heatMap2dWavefunction, abs2, fromHartoEv,
         makeJustAnother2DgraphComplexSINGLE, fromLabelsToFloats, derivative2dGamTheMu,
         graphic_Pulse,derivative3dMu,equilibriumIndex, readWholeH5toDict, err)
from quantumpropagator.CPropagator import (CextractEnergy3dMu, Cderivative3dMu, Cenergy_2d_GamThe,
                                           Cderivative_2d_GamThe,Cenergy_1D_Phi, Cderivative_1D_Phi,
                                           Cenergy_1D_Gam, Cderivative_1D_Gam, Cenergy_1D_The,
                                           Cderivative_1D_The,Crk4Ene3d, version_Cpropagator, pulZe)

def calculate_stuffs_on_WF(single_wf, inp, outputFile):
    '''
    This function is a standalone function that recreates the output file counting also the absorbing potential
    '''
    counter = 0
    nstates = inp['nstates']
    ii = 0
    wf = single_wf['WF']
    t_fs,t = single_wf['Time']
    kind = inp['kind']
    if kind != '3d':
        err('This function is implemented only in 3d code')
    CEnergy, Cpropagator = select_propagator(kind)
    kin, pot, pul, absS = CEnergy(t,wf,inp)
    kinetic = np.vdot(wf,kin)
    potential = np.vdot(wf,pot)
    pulse_interaction = np.vdot(wf,pul)
    absorbing_potential = np.vdot(wf,absS)
    absorbing_potential_thing = np.real(-2j * absorbing_potential)
    total = kinetic + potential + pulse_interaction
    initialTotal = inp['initialTotal']
    norm_wf = np.linalg.norm(wf)

    outputStringS = ' {:04d} |{:10d} |{:11.4f} | {:+e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+10.3e} | {:+10.3e} | {:+10.3e} | {:+10.3e} |'
    outputString = outputStringS.format(counter, ii,t*0.02418884,1-norm_wf,fromHartoEv(kinetic.real),fromHartoEv(potential.real),fromHartoEv(total.real),fromHartoEv(initialTotal - total.real), fromHartoEv(pulse_interaction.real), pulZe(t,inp['pulseX']), pulZe(t,inp['pulseY']), pulZe(t,inp['pulseZ']), absorbing_potential_thing)
    print(outputString)

    kind = inp['kind']
    outputString_abs = "{:11.4f} {:+7.5e}".format(t,absorbing_potential_thing)
    for i in range(nstates):
        if kind == '3d':
            singleStatewf = wf[:,:,:,i]
            singleAbspote = absS[:,:,:,i]
            norm_loss_this_step = np.real(-2j * np.vdot(singleStatewf,singleAbspote))
        if kind == 'GamThe':
            err('no 2d here')
        elif kind == 'Phi' or kind == 'Gam' or kind == 'The':
            err('no 1d here')

        outputString_abs += " {:+7.5e} ".format(norm_loss_this_step)

    with open(outputFile, "a") as oof:
        outputStringS2 = '{}'
        outputString2 = outputStringS2.format(outputString_abs)
        oof.write(outputString2 + '\n')



def expandcube(inp):
    return inp

def doubleAxespoins1(Y):
    N = len(Y)
    X = np.arange(0, 2*N, 2)
    X_new = np.arange(2*N-1)       # Where you want to interpolate
    Y_new = np.interp(X_new, X, Y)
    return(Y_new)

def doubleAxespoins(Y):
    return (doubleAxespoins1(doubleAxespoins1(Y)))

def select_propagator(kind):
    '''
    This function will return correct function name for the propagators
    kind :: String <- the kind of dynamics
    '''
    Propagators = {'3d'     : (CextractEnergy3dMu, Cderivative3dMu),
                   'GamThe' : (Cenergy_2d_GamThe,Cderivative_2d_GamThe),
                   'Phi'    : (Cenergy_1D_Phi, Cderivative_1D_Phi),
                   'Gam'    : (Cenergy_1D_Gam, Cderivative_1D_Gam),
                   'The'    : (Cenergy_1D_The, Cderivative_1D_The),
                  }
    return Propagators[kind]

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
        warning('WF widened using factor: {}'.format(factor))
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

    if 'initial_state' in inputDict:
        initial_state = inputDict['initial_state']
        warning('Initial gaussian wavepacket in state {}'.format(initial_state))
    else:
        initial_state = 0

    wf = np.zeros((phiL, gamL, theL, nstates), dtype=complex)

    print(initial_state)
    wf[:,:,:,initial_state] = initialCondition3d(wf[:,:,:,initial_state],dataDict,factor,displ,init_mom)

    # Take values array from labels (radians already)
    phis,gams,thes = fromLabelsToFloats(dataDict)

    # take step
    dphi = phis[0] - phis[1]
    dgam = gams[0] - gams[1]
    dthe = thes[0] - thes[1]

    inp = { 'dt'       : inputDict['dt'],
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
            'nacCube'  : dataDict['smoCube'],
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

    if 'multiply_nac' in inputDict:
        nac_multiplier = inputDict['multiply_nac']
        warning('Nac are multiplied by {}'.format(nac_multiplier))
        inp['nacCube'] = inp['nacCube'] * nac_multiplier


    nameRoot = create_enumerated_folder(inputDict['outFol'])
    inputDict['outFol'] = nameRoot
    inp['outFol'] = nameRoot
    numStates = inputDict['states']

    ###################
    # Absorbing Thing #
    ###################

    if 'absorb' in inputDict:
        good('ABSORBING POTENTIAL is taken from file')
        file_absorb = inputDict['absorb']
        print('{}'.format(file_absorb))
        inp['absorb'] = retrieve_hdf5_data(file_absorb,'absorb')
    else:
        good('NO ABSORBING POTENTIAL')
        inp['absorb'] = np.zeros_like(inp['potCube'])


    ################
    # slice states #
    ################

    kind = inp['kind']

    # Take equilibrium points from directionFile
    # warning('This is a bad equilibriumfinder')
    # gsm_phi_ind, gsm_gam_ind, gsm_the_ind = equilibriumIndex(inputDict['directions1'],dataDict)
    gsm_phi_ind, gsm_gam_ind, gsm_the_ind = (29,28,55)
    warning('You inserted equilibrium points by hand: {} {} {}'.format(gsm_phi_ind, gsm_gam_ind, gsm_the_ind))

    inp['nstates'] = numStates
    if kind == '3d':
        inp['potCube'] = inp['potCube'][:,:,:,:numStates]
        inp['kinCube'] = inp['kinCube'][:,:,:]
        inp['dipCube'] = inp['dipCube'][:,:,:,:,:numStates,:numStates]
        wf =                         wf[:,:,:,:numStates]
        good('Propagation in 3D.')
        print('\nDimensions:\nPhi: {}\nGam: {}\nThet: {}\nNstates: {}\nNatoms: {}\n'.format(phiL, gamL, theL,numStates, natoms))
    elif kind == 'GamThe':
        inp['potCube'] = inp['potCube'][gsm_phi_ind,:,:,:numStates]
        inp['kinCube'] = inp['kinCube'][gsm_phi_ind,:,:]
        inp['dipCube'] = inp['dipCube'][gsm_phi_ind,:,:,:,:numStates,:numStates]
        wf             =             wf[gsm_phi_ind,:,:,:numStates]
        good('Propagation in GAM-THE with Phi {}'.format(gsm_phi_ind))
        print('Shapes: P:{} K:{} W:{} D:{}'.format(inp['potCube'].shape, inp['kinCube'].shape, wf.shape, inp['dipCube'].shape))
        print('\nDimensions:\nGam: {}\nThe: {}\nNstates: {}\nNatoms: {}\n'.format(gamL, theL, numStates, natoms))
        norm_wf = np.linalg.norm(wf)
        wf = wf / norm_wf
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

        sposta = False
        if sposta:
            gsm_phi_ind = 20
            gsm_gam_ind = 20
            warning('Phi is {}, NOT EQUILIBRIUM'.format(gsm_phi_ind))
            warning('Gam is {}, NOT EQUILIBRIUM'.format(gsm_gam_ind))

        inp['potCube'] = inp['potCube'][gsm_phi_ind,gsm_gam_ind,:,:numStates]
        inp['absorb']  = inp['absorb'][gsm_phi_ind,gsm_gam_ind,:,:numStates]
        inp['kinCube'] = inp['kinCube'][gsm_phi_ind,gsm_gam_ind,:]
        inp['dipCube'] = inp['dipCube'][gsm_phi_ind,gsm_gam_ind,:,:,:numStates,:numStates]
        inp['nacCube'] = inp['nacCube'][gsm_phi_ind,gsm_gam_ind,:,:numStates,:numStates,:]
        wf             =             wf[gsm_phi_ind,gsm_gam_ind,:,:numStates]

        #  doubleGridPoints
        doubleThis = False
        if doubleThis:
            warning('POINTS DOUBLED ALONG THETA')
            inp['thes'] = doubleAxespoins(inp['thes'])
            inp['theL'] = inp['thes'].size
            inp['dthe'] = inp['thes'][0] - inp['thes'][1]
            inp['potCube'] = np.array([doubleAxespoins(x) for x in inp['potCube'].T]).T

            newWf =  np.empty((inp['theL'],numStates), dtype=complex)
            for ssssss in range(numStates):
                newWf[:,ssssss] = doubleAxespoins(wf[:,ssssss])

            wf = newWf

            newNac = np.empty((inp['theL'],numStates,numStates,3))
            for nnn in range(2):
                for mmm in range(2):
                    for aaa in range(3):
                        newNac[:,nnn,mmm,aaa] = doubleAxespoins(inp['nacCube'][:,nnn,mmm,aaa])
            inp['nacCube'] = newNac

            newKin = np.empty((inp['theL'],9,3))
            for nnn in range(9):
                for mmm in range(3):
                    newKin[:,nnn,mmm] = doubleAxespoins(inp['kinCube'][:,nnn,mmm])
            inp['kinCube'] = newKin


        good('Propagation in THE with Phi {} and Gam {}'.format(gsm_phi_ind,gsm_gam_ind))
        print('Shapes: P:{} K:{} W:{} D:{}'.format(inp['potCube'].shape, inp['kinCube'].shape, wf.shape, inp['dipCube'].shape))
        print('\nDimensions:\nThe: {}\nNstates: {}\nNatoms: {}\n'.format(theL, numStates, natoms))
        norm_wf = np.linalg.norm(wf)
        wf = wf / norm_wf
    else:
        err('I do not recognize the kind')

    # take a wf from file (and not from initial condition)
    if 'initialFile' in inputDict:
        warning('we are taking initial wf from file')
        wffn = inputDict['initialFile']
        print('File -> {}'.format(wffn))
        wf_not_norm = retrieve_hdf5_data(wffn,'WF')
        wf = wf_not_norm/np.linalg.norm(wf_not_norm)

    #############################
    # PROPAGATOR SELECTION HERE #
    #############################

    CEnergy, Cpropagator = select_propagator(kind)
    good('Cpropagator version: {}'.format(version_Cpropagator()))

    # INITIAL DYNAMICS VALUES
    dt = inp['dt']
    t = 0
    counter  = 0
    fulltime = inp['fullTime']
    fulltimeSteps = int(fulltime/dt)
    deltasGraph = inputDict['deltasGraph']
    print('I will do {} steps.\n'.format(fulltimeSteps))
    outputFile = os.path.join(nameRoot, 'output')
    outputFileP = os.path.join(nameRoot, 'outputPopul')
    outputFileA = os.path.join(nameRoot, 'output_Absorbing')
    print('\ntail -f {}\n'.format(outputFileP))

    # calculating initial total/potential/kinetic
    kin, pot, pul, absS = CEnergy(t,wf,inp)
    kinetic = np.vdot(wf,kin)
    potential = np.vdot(wf,pot)
    pulse_interaction = np.vdot(wf,pul)
    initialTotal = kinetic + potential + pulse_interaction
    inp['initialTotal'] = initialTotal.real

    # to give the graph a nice range
    inp['vmax_value'] = abs2(wf).max()

    # graph the pulse
    graphic_Pulse(inp)

    # saving input data in h5 file
    dataH5filename = os.path.join(nameRoot, 'allInput.h5')
    writeH5fileDict(dataH5filename,inp)

    # print top of table
    header = ' Coun |  step N   |       fs   |  NORM devia.  | Kin. Energy  | Pot. Energy  | Total Energy | Tot devia.   | Pulse_Inter. |  Pulse X   |  Pulse Y   |  Pulse Z   |  Norm Loss |'
    bar = ('-' * (len(header)))
    print('Energies in ElectronVolt \n{}\n{}\n{}'.format(bar,header,bar))

    for ii in range(fulltimeSteps):
        if (ii % deltasGraph) == 0 or ii==fulltimeSteps-1:
            #  async is awesome. But it is not needed in 1d and maybe in 2d.
            if kind == '3D':
                asyncFun(doAsyncStuffs,wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy)
            else:
                doAsyncStuffs(wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy)
            counter += 1

        wf = Crk4Ene3d(Cpropagator,t,wf,inp)
        t  = t + dt

def restart_propagation(inp,inputDict):
    '''
    This function restarts a propagation that has been stopped
    '''
    import glob

    nameRoot = inputDict['outFol']
    list_wave_h5 = sorted(glob.glob(nameRoot + '/Gaussian*.h5'))
    last_wave_h5 = list_wave_h5[-1]
    wf = retrieve_hdf5_data(last_wave_h5,'WF')
    t = retrieve_hdf5_data(last_wave_h5,'Time')[1] # [1] is atomic units
    kind = inp['kind']
    deltasGraph = inputDict['deltasGraph']
    counter = len(list_wave_h5) - 1

    dt = inputDict['dt']
    fulltime = inputDict['fullTime']

    fulltimeSteps = int(fulltime/dt)
    outputFile = os.path.join(nameRoot, 'output')
    outputFileP = os.path.join(nameRoot, 'outputPopul')

    if (inputDict['fullTime'] == inp['fullTime']):
        good('Safe restart with same fulltime')
        #h5_data_file = os.path.join(nameRoot,'allInput.h5')
    else:
        h5_data_file = os.path.join(nameRoot,'allInput.h5')
        dict_all_data = readWholeH5toDict(h5_data_file)
        dict_all_data['fullTime'] = inputDict['fullTime']
        writeH5fileDict(h5_data_file, dict_all_data)
        good('different fullTime detected and allInput updated')

    print('\ntail -f {}\n'.format(outputFileP))
    CEnergy, Cpropagator = select_propagator(kind)
    good('Cpropagator version: {}'.format(version_Cpropagator()))

    ii_initial = counter * deltasGraph
    print('I will do {} more steps.\n'.format(fulltimeSteps-ii_initial))

    if True:
        print('Calculation restart forced on me... I assume you did everything you need')
    else:
        warning('Did you restart this from a finished calculation?')
        strout = "rm {}\nsed -i '$ d' {}\nsed -i '$ d' {}\n"
        print(strout.format(last_wave_h5,outputFile,outputFileP))
        input("Press Enter to continue...")

    strOUT = '{} {} {}'.format(ii_initial,counter,fulltimeSteps)
    good(strOUT)
    for ii in range(ii_initial,fulltimeSteps):
        #print('ii = {}'.format(ii))
        if ((ii % deltasGraph) == 0 or ii==fulltimeSteps-1):
                #  async is awesome. But it is not needed in 1d and maybe in 2d.
                if kind == '3D':
                    asyncFun(doAsyncStuffs,wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy)
                else:
                    doAsyncStuffs(wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy)
                counter += 1

        wf = Crk4Ene3d(Cpropagator,t,wf,inp)
        t  = t + dt


def doAsyncStuffs(wf,t,ii,inp,inputDict,counter,outputFile,outputFileP,CEnergy):
    nameRoot = inputDict['outFol']
    nstates = inp['nstates']
    name = os.path.join(nameRoot, 'Gaussian' + '{:04}'.format(counter))
    h5name = name + ".h5"
    writeH5file(h5name,[("WF", wf),("Time", [t*0.02418884,t])])
    kin, pot, pul, absS = CEnergy(t,wf,inp)

    kinetic = np.vdot(wf,kin)
    potential = np.vdot(wf,pot)
    pulse_interaction = np.vdot(wf,pul)
    absorbing_potential = np.vdot(wf,absS)
    # you asked and discussed with Stephan about it. This is the norm loss due to CAP complex absorbing potential. It needs to be multiplied by -2i.
    absorbing_potential_thing = np.real(-2j * absorbing_potential)
    total = kinetic + potential + pulse_interaction
    initialTotal = inp['initialTotal']
    norm_wf = np.linalg.norm(wf)

    ##  you wanted to print the header when the table goes off screen... this is why you get the rows number
    #rows, _ = os.popen('stty size', 'r').read().split()
    #if int(rows) // counter == 0:
    #    print('zero')

    outputStringS = ' {:04d} |{:10d} |{:11.4f} | {:+e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+7.5e} | {:+10.3e} | {:+10.3e} | {:+10.3e} | {:+10.3e} |'
    outputString = outputStringS.format(counter, ii,t*0.02418884,1-norm_wf,fromHartoEv(kinetic.real),fromHartoEv(potential.real),fromHartoEv(total.real),fromHartoEv(initialTotal - total.real), fromHartoEv(pulse_interaction.real), pulZe(t,inp['pulseX']), pulZe(t,inp['pulseY']), pulZe(t,inp['pulseZ']), absorbing_potential_thing)
    print(outputString)

    kind = inp['kind']
    outputStringSP = "{:11.4f}".format(t/41.3)
    for i in range(nstates):
        if kind == '3d':
            singleStatewf = wf[:,:,:,i]
        if kind == 'GamThe':
            singleStatewf = wf[:,:,i]
        elif kind == 'Phi' or kind == 'Gam' or kind == 'The':
            singleStatewf = wf[:,i]
        outputStringSP += " {:+7.5e} ".format(np.linalg.norm(singleStatewf)**2)


    with open(outputFileP, "a") as oofP:
        oofP.write(outputStringSP + '\n')

    with open(outputFile, "a") as oof:
        outputStringS2 = '{} {} {} {} {} {} {} {} {} {} {} {}'
        outputString2 = outputStringS2.format(counter,ii,t/41.3,1-norm_wf,fromHartoEv(kinetic.real),fromHartoEv(potential.real),fromHartoEv(total.real),fromHartoEv(initialTotal - total.real), pulZe(t,inp['pulseX']), pulZe(t,inp['pulseY']), pulZe(t,inp['pulseZ']), absorbing_potential_thing)
        oof.write(outputString2 + '\n')

    #####################
    # on the fly graphs #
    #####################

    if 'graphs' in inputDict:
        vmaxV = inp['vmax_value']
        # I am sure there is a better way to do this...
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
        if kind == 'Phi' or kind == 'Gam' or kind == 'The':
            graphFileName = name + ".png"
            makeJustAnother2DgraphComplexSINGLE(valuesX,wf,graphFileName,label)

        if kind == 'GamThe':
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
    #gsm_phi_ind = dataDict['phis'].index('P000-000')
    #gsm_gam_ind = dataDict['gams'].index('P016-923')
    #gsm_the_ind = dataDict['thes'].index('P114-804')

    gsm_phi_ind = 29
    gsm_gam_ind = 28
    gsm_the_ind = 55

    warning('Equilibrium points put by hand: {} {} {}'.format(gsm_phi_ind,gsm_gam_ind,gsm_the_ind))

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
    coe_gam = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,4,2]
    # these three lines are here because we wanted to debug Gamma
    #coe_gam = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,0,2]
    #warning('coe_gam has been changed !!! in initialcondition function')
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

