''' this is the module for the hamiltonian '''

import numpy as np
import os
from quantumpropagator import (printDict, printDictKeys, loadInputYAML, bring_input_to_AU,
         warning, labTranformA, gaussian2, makeJustAnother2DgraphComplex,
         fromHartreetoCmMin1, makeJustAnother2DgraphMULTI,derivative3d,rk4Ene3d,derivative1dPhi,
         good, asyncFun, derivative1dGam, create_enumerated_folder,
         makeJustAnother2DgraphComplexALLS)

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
    print('{} {} {} {} {}'.format(phiL, gamL, theL, nstates, natoms))

    # INITIAL WF
    wf = np.zeros((phiL, gamL, theL), dtype=complex)
    wf = initialCondition3d(wf,dataDict)

    # Take values array from labels
    phis = labTranformA(dataDict['phis'])
    gams = labTranformA(dataDict['gams'])
    thes = labTranformA(dataDict['thes'])

    # take step
    dphi = phis[0] - phis[1]
    dgam = gams[0] - gams[1]
    dthe = thes[0] - thes[1]

    # INITIAL DYNAMICS VALUES
    h = inputDict['dt']
    t = 0
    counter  = 0
    fulltime = 20
    deltasGraph = 50


    inp = { 'h'      : h,
            'phiL'   : phiL,
            'gamL'   : gamL,
            'theL'   : theL,
            'natoms' : natoms,
            'phis'   : phis,
            'gams'   : gams,
            'thes'   : thes,
            'dphi'   : dphi,
            'dgam'   : dgam,
            'dthe'   : dthe,
            'potCube': dataDict['potCube'],
            'kinCube': dataDict['kinCube'],
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

    nameRoot = inputDict['outFol']
    folder = create_enumerated_folder(nameRoot)
    inputDict['outFol'] = folder

    for ii in range(fulltime):
        # propagation in phi only
        #wf = rk4Ene3d(derivative1dPhi,t,wf,inp)

        # propagation in gam only
        #wf = rk4Ene3d(derivative1dGam,t,wf,inp)

        # propagation in 3d
        wf = rk4Ene3d(derivative3d,t,wf,inp)
        #print('WF: {}'.format(wf))
        norm_wf = np.linalg.norm(wf)
        good('NORM deviation : {}'.format(1-norm_wf))
        if (ii % deltasGraph) == 0:
            name = os.path.join(nameRoot, 'Gaussian' + '{:04}'.format(counter))
            counter += 1
            #asyncFun(makeJustAnother2DgraphComplexALLS, phis, np.array([wf]), name,"gauss " + '{:8.5f}'.format(t/41.5), xaxisL=[-10,10])


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


def initialCondition3d(wf,dataDict):
    '''
    calculates the initial condition WV
    wf :: np.array(phiL,gamL,theL)  Complex
    datadict :: Dictionary {}
    '''

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

    print('{} {} {}'.format(dphi,dgam,dthe))

    # slice out the parabolas at equilibrium geometry
    pot = dataDict['potCube']
    parabola_phi = pot[:,gsm_gam_ind,gsm_the_ind,0]
    parabola_gam = pot[gsm_phi_ind,:,gsm_the_ind,0]
    parabola_the = pot[gsm_phi_ind,gsm_gam_ind,:,0]

    # calculate force with finite difference
    force_phi = forcehere(parabola_phi, gsm_phi_ind, h=dphi)
    force_gam = forcehere(parabola_gam, gsm_gam_ind, h=dgam)
    force_the = forcehere(parabola_the, gsm_the_ind, h=dthe)

    print('{} {} {}'.format(force_phi,force_gam,force_the))

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
    factor = 1
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

    phi0 = phis[gsm_phi_ind]
    gam0 = gams[gsm_gam_ind]
    the0 = thes[gsm_the_ind]

    for p, phi in enumerate(phis):
        phiV = gaussian2(phi, phi0, Gw_phi)
        for g, gam in enumerate(gams):
            gamV = gaussian2(gam, gam0, Gw_gam)
            for t , the in enumerate(thes):
                theV = gaussian2(the, the0, Gw_the)
                #print('I: {}\tV: {}\tZ: {}\tG: {}\t'.format(t,the,the0,theV))

                wf[p,g,t] = phiV * gamV * theV

    norm_wf = np.linalg.norm(wf)
    print('NORM: {}'.format(norm_wf))
    wf = wf / norm_wf
    print(wf.shape)
    print('\n\nk: {} {} {}'.format(force_phi,force_gam,force_the))
    print('coe: {} {} {}'.format(coe_phi,coe_gam,coe_the))
    print('G: {} {} {}'.format(G_phi,G_gam,G_the))
    print('Gw: {} {} {}'.format(Gw_phi,Gw_gam,Gw_the))
    print('w: {} {} {}'.format(w_phi,w_gam,w_the))
    print('cm-1: {} {} {}'.format(fromHartreetoCmMin1(w_phi),
                                  fromHartreetoCmMin1(w_gam),
                                  fromHartreetoCmMin1(w_the)))
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



