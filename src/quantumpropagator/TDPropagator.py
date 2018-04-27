''' this is the module for the hamiltonian '''

import numpy as np
import os
from quantumpropagator import (printDict, printDictKeys, loadInputYAML, bring_input_to_AU,
         warning, labTranformA, gaussian2, makeJustAnother2DgraphComplex,
         fromHartreetoCmMin1, makeJustAnother2DgraphMULTI,derivative3d,rk4Ene3d,derivative1dPhi,
         good, asyncFun, derivative1dGam, create_enumerated_folder, fromCmMin1toFs,
         makeJustAnother2DgraphComplexALLS, derivative2dGamThe, retrieve_hdf5_data,
         writeH5file, writeH5fileDict)
from quantumpropagator.CPropagator import Cderivative2dGamThe

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

    # INITIAL WF
    factor = 500
    wf = np.zeros((phiL, gamL, theL), dtype=complex)
    wf = initialCondition3d(wf,dataDict,factor)

    # Take values array from labels
    phis = labTranformA(dataDict['phis'])
    gams = labTranformA(dataDict['gams'])
    thes = labTranformA(dataDict['thes'])

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
            }

    ## REDUCE THE PROBLEM IN 1D 1 state
    ## Take equilibrium points
    #gsm_phi_ind = dataDict['phis'].index('P000-000')
    #gsm_gam_ind = dataDict['gams'].index('P016-211')
    #gsm_the_ind = dataDict['thes'].index('P114-719')

    #zero_pot = dataDict['potCube'] - np.amin(dataDict['potCube'])
    #inp['potCube'] = zero_pot[:,gsm_gam_ind,gsm_the_ind,0]
    #inp['kinCube'] = dataDict['kinCube'][:,gsm_gam_ind,gsm_the_ind]
    #wf             =       groundStateWF[:,gsm_gam_ind,gsm_the_ind]
    #print('shapes: P:{} K:{} W:{} '.format(inp['potCube'].shape,inp['kinCube'].shape,wf.shape))
    #norm_wf = np.linalg.norm(wf)
    #wf = wf / norm_wf


    inp['potCube'] = dataDict['potCube'] - np.amin(dataDict['potCube'])
    norm_wf = np.linalg.norm(wf)
    good('starting NORM deviation : {}'.format(1-norm_wf))

    counter = 0

    nameRoot = create_enumerated_folder(inputDict['outFol'])
    inputDict['outFol'] = nameRoot

    ## REDUCE THE PROBLEM IN 2d THE GAM
    ## Take equilibrium points
    gsm_phi_ind = dataDict['phis'].index('P000-000')
    gsm_gam_ind = dataDict['gams'].index('P016-211')
    gsm_the_ind = dataDict['thes'].index('P114-719')

    inp['potCube'] = inp['potCube'][gsm_phi_ind,:,:]
    inp['kinCube'] = inp['kinCube'][gsm_phi_ind,:,:]
    wf =                         wf[gsm_phi_ind,:,:]

    # take a wf from file (and not from initial condition)
    externalFile = True
    if externalFile:
        warning('we are taking initial wf from file')
        wffn = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/GaussianMoved.h5'
        #wffn = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/Gaussian0001.h5'
        print(wffn)
        wf_not_norm = retrieve_hdf5_data(wffn,'WF')
        wf = wf_not_norm/np.linalg.norm(wf_not_norm)

    # INITIAL DYNAMICS VALUES
    h = inp['h']
    t = 0
    counter  = 0
    fulltime = inp['fullTime']
    fulltimeSteps = int(fulltime/h)
    deltasGraph = inputDict['deltasGraph']
    print('Dimensions:\nPhi: {}\nGam: {}\nThet: {}\nNstates: {}\nNatoms: {}'.format(phiL, gamL, theL, nstates, natoms))
    print('I will do {} steps.\n'.format(fulltimeSteps))
    outputFile = os.path.join(nameRoot, 'output')

    # saving input data in h5 file
    dataH5filename = os.path.join(nameRoot, 'allInput.h5')
    writeH5fileDict(dataH5filename,inp)

    header = '  step N   |      fs   |      NORM     | Total Energy | Tot deviation'
    bar = ('-' * (len(header)))
    print('{}\n{}\n{}'.format(bar,header,bar))

    for ii in range(fulltimeSteps):
        dPsiDt = Cderivative2dGamThe

        # propagation in Gam The


        if (ii % deltasGraph) == 0 or ii==fulltimeSteps-1:
            name = os.path.join(nameRoot, 'Gaussian' + '{:04}'.format(counter))
            counter += 1
            h5name = name + ".h5"
            asyncFun(writeH5file,h5name,[("WF", wf),("Time", [t/41.5,t])])
            #print('Here I calculate total energy:')
            tot = dPsiDt(t,wf,inp) * 1j
            total = np.vdot(wf,tot)
            if ii == 0:
                initialTotal = total.real
            norm_wf = np.linalg.norm(wf)
            outputStringS = '{:10d} |{:10.4f} | {:+e} | {:+7.5e} | {:+7.5e}'
            outputString = outputStringS.format(ii,t/41.3,1-norm_wf,total.real,initialTotal - total.real)
            print(outputString)
            with open(outputFile, "a") as oof:
                outputStringS2 = '{} {} {} {} {}'
                outputString2 = outputStringS2.format(ii,t/41.3,1-norm_wf,total.real,initialTotal - total.real)
                oof.write(outputString2 + '\n')

        wf = rk4Ene3d(dPsiDt,t,wf,inp)
        t  = t + h

    print('\n\n\n')

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


def initialCondition3d(wf,dataDict,factor=None):
    '''
    calculates the initial condition WV
    wf :: np.array(phiL,gamL,theL)  Complex
    datadict :: Dictionary {}
    '''
    good('Initial condition printing')

    # Take equilibrium points
    gsm_phi_ind = dataDict['phis'].index('P000-000')
    gsm_gam_ind = dataDict['gams'].index('P016-211')
    gsm_the_ind = dataDict['thes'].index('P114-719')

    # Take values array from labels
    phis = labTranformA(dataDict['phis'])
    gams = labTranformA(dataDict['gams'])
    thes = labTranformA(dataDict['thes'])

    # take step
    dphi = phis[0] - phis[1]
    dgam = gams[0] - gams[1]
    dthe = thes[0] - thes[1]


    # slice out the parabolas at equilibrium geometry
    pot = dataDict['potCube']
    parabola_phi = pot[:,gsm_gam_ind,gsm_the_ind,0]
    parabola_gam = pot[gsm_phi_ind,:,gsm_the_ind,0]
    parabola_the = pot[gsm_phi_ind,gsm_gam_ind,:,0]

    # calculate force with finite difference
    force_phi = forcehere(parabola_phi, gsm_phi_ind, h=dphi)
    force_gam = forcehere(parabola_gam, gsm_gam_ind, h=dgam)
    force_the = forcehere(parabola_the, gsm_the_ind, h=dthe)

    # Now, I want the coefficients of the second derivative of the kinetic energy jacobian
    # for the equilibrium geometry, so that I can calculate the gaussian.
    # in the diagonal approximation those are the diagonal elements, thus element 0,4,8.

    coe_phi = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,0,2]
    coe_gam = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,4,2]
    coe_the = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,8,2]

    # they need to be multiplied by (-2 * hbar**2), where hbar is 1. And inverted
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
    displPhi = 0
    displGam = -5
    displThe = 9
    if displPhi != 0 or displGam != 0 or displThe != 0:
       warning('Some displacements activated | Phi {} | Gam {} | The {}'.format(displPhi,displGam,displThe))

    phi0 = phis[gsm_phi_ind + displPhi]
    gam0 = gams[gsm_gam_ind + displGam]
    the0 = thes[gsm_the_ind + displThe]

    for p, phi in enumerate(phis):
        phiV = gaussian2(phi, phi0, Gw_phi)
        for g, gam in enumerate(gams):
            gamV = gaussian2(gam, gam0, Gw_gam)
            for t , the in enumerate(thes):
                theV = gaussian2(the, the0, Gw_the)
                #print('I: {}\tV: {}\tZ: {}\tG: {}\t'.format(t,the,the0,theV))

                wf[p,g,t] = phiV * gamV * theV

    norm_wf = np.linalg.norm(wf)
    print('NORM before normalization: {:e}'.format(norm_wf))
    print('Steps: phi({:.3f}) gam({:.3f}) the({:.3f})'.format(dphi,dgam,dthe))
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

    #wf_phi_line = wf[:,gsm_gam_ind,gsm_the_ind]
    #wf_gam_line = wf[gsm_phi_ind,:,gsm_the_ind]
    #wf_the_line = wf[gsm_phi_ind,gsm_gam_ind,:]

    #mja2gc = makeJustAnother2DgraphComplex
    #mja2gc(phis,wf_phi_line,'gau_phi','phi')
    #mja2gc(gams,wf_gam_line,'gau_gam','gam')
    #mja2gc(thes,wf_the_line,'gau_the','the')

    ##graph these parabolas
    #mja2g = makeJustAnother2Dgraph
    #mja2g('par1','phi',parabola1)
    #mja2g('par2','gamma',parabola2)
    #mja2g('par3','theta',parabola3)

    # graph the parabola
    #mja2dgm = makeJustAnother2DgraphMULTI
    #zero_par_phi = parabola_phi - np.amin(parabola_phi)
    #par_force_phi = force_phi * (phis-phi0)**2
    #toSee_phi = np.stack((wf_phi_line,zero_par_phi,par_force_phi),axis=1)
    #mja2dgm(phis,toSee_phi,'parpot_phi','phi')

    #zero_par_gam = parabola_gam - np.amin(parabola_gam)
    #par_force_gam = force_gam * (gams-gam0)**2
    #toSee_gam = np.stack((wf_gam_line,zero_par_gam,par_force_gam),axis=1)
    #mja2dgm(gams,toSee_gam,'parpot_gam','gam')

    #zero_par_the = parabola_the - np.amin(parabola_the)
    #par_force_the = force_the * (thes-the0)**2
    #toSee_the = np.stack((wf_the_line,zero_par_the,par_force_the),axis=1)
    #mja2dgm(thes,toSee_the,'parpot_the','the')

    return(wf)


if __name__ == "__main__":
    fn1 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/input.yml'
    fn2 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/datainput.npy'
    inputDict = bring_input_to_AU(loadInputYAML(fn1))
    dataDict = np.load(fn2) # this is a numpy wrapper, for this we use [()]
    propagate3D(dataDict[()], inputDict)



