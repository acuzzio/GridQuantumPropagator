''' this is the module for the hamiltonian '''

import numpy as np
from quantumpropagator import (printDict, printDictKeys, loadInputYAML, bring_input_to_AU,
         makeJustAnother2Dgraph, warning, labTranformA, gaussian2, makeJustAnother2DgraphComplex)

def propagate3D(dataDict, inputDict):
    '''
    Two dictionaries, one from data file and one from input file
    it starts and run the 3d propagation of the wavefunction...
    '''
    printDict(inputDict)
    printDictKeys(dataDict)
    printDictKeys(inputDict)

    h = inputDict['dt']
    startState = inputDict['states']
    _, _, _, nstates = dataDict['potCube'].shape
    phiL, gamL, theL, natoms, _ = dataDict['geoCUBE'].shape
    print('{} {} {} {} {}'.format(phiL, gamL, theL, nstates, natoms))
    wf = np.zeros((phiL, gamL, theL), dtype=complex)
    groundStateWF = initialCondition3d(wf,dataDict)
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
    gsm_phi_ind = dataDict['phis'].index('P000-000')
    gsm_gam_ind = dataDict['gams'].index('P016-211')
    gsm_the_ind = dataDict['thes'].index('P114-719')

    phis = labTranformA(dataDict['phis'])
    gams = labTranformA(dataDict['gams'])
    thes = labTranformA(dataDict['thes'])

    dphi = phis[0] - phis[1]
    dgam = gams[0] - gams[1]
    dthe = thes[0] - thes[1]
    print('{} {} {}'.format(dphi,dgam,dthe))

    # slice out the parabolas
    pot = dataDict['potCube']
    parabola_phi = pot[:,gsm_gam_ind,gsm_the_ind,0]
    parabola_gam = pot[gsm_phi_ind,:,gsm_the_ind,0]
    parabola_the = pot[gsm_phi_ind,gsm_gam_ind,:,0]

    force_phi = forcehere(parabola_phi, gsm_phi_ind, h=dphi)
    force_gam = forcehere(parabola_gam, gsm_gam_ind, h=dgam)
    force_the = forcehere(parabola_the, gsm_the_ind, h=dthe)
    print('{} {} {}'.format(force_phi,force_gam,force_the))

    # Now, I want the coefficients of the second derivative of the kinetic energy jacobian
    # for the equilibrium geometry, so that I can calculate the gaussian.
    # in the diagonal approximation those are the diagonal elements, thus element 0,4,8.
    # they need to be multiplied by (-2 * hbar**2), where hbar is 1.

    coe_phi = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,0,2]
    coe_gam = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,4,2]
    coe_the = dataDict['kinCube'][gsm_phi_ind,gsm_gam_ind,gsm_the_ind,8,2]
    mass_phi = -2 * (1/coe_phi)
    mass_gam = -2 * (1/coe_gam)
    mass_the = -2 * (1/coe_the)

    # MAYBE ERROR - 
    # mw = np.sqrt(Gk)  
    # ???
    Gw_phi = np.sqrt(force_phi*mass_phi)
    Gw_gam = np.sqrt(force_gam*mass_gam)
    Gw_the = np.sqrt(force_the*mass_the)

    w_phi = np.sqrt(force_phi/mass_phi)
    w_gam = np.sqrt(force_gam/mass_gam)
    w_the = np.sqrt(force_the/mass_the)

    for p, phi in enumerate(phis):
        phi0 = phis[gsm_phi_ind]
        phiV = gaussian2(phi, phi0, Gw_phi)
        for g, gam in enumerate(gams):
            gam0 = gams[gsm_gam_ind]
            gamV = gaussian2(gam, gam0, Gw_gam)
            for t , the in enumerate(thes):
                the0 = thes[gsm_the_ind]
                theV = gaussian2(the, the0, Gw_the)
                print('I: {}\tV: {}\tZ: {}\tG: {}\t'.format(t,the,the0,theV))

                wf[p,g,t] = phiV * gamV * theV

    norm_wf = np.linalg.norm(wf)
    wf = wf / norm_wf
    print(wf.shape)
    print('k: {} {} {}'.format(force_phi,force_gam,force_the))
    print('coe: {} {} {}'.format(coe_phi,coe_gam,coe_the))
    print('G: {} {} {}'.format(mass_phi,mass_gam,mass_the))
    print('Gw: {} {} {}'.format(Gw_phi,Gw_gam,Gw_the))
    print('w: {} {} {}'.format(w_phi,w_gam,w_the))
    makeJustAnother2DgraphComplex(phis,wf[:,gsm_gam_ind,gsm_the_ind],'gau_phi','phi')
    makeJustAnother2DgraphComplex(gams,wf[gsm_phi_ind,:,gsm_the_ind],'gau_gam','gam')
    makeJustAnother2DgraphComplex(thes,wf[gsm_phi_ind,gsm_gam_ind,:],'gau_the','the')





    ##graph these parabolas 
    #makeJustAnother2Dgraph('par1','phi',parabola1)
    #makeJustAnother2Dgraph('par2','gamma',parabola2)
    #makeJustAnother2Dgraph('par3','theta',parabola3)


if __name__ == "__main__":
    fn1 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/hugo.yml'
    fn2 = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/datahugo.npy'
    inputDict = bring_input_to_AU(loadInputYAML(fn1))
    dataDict = np.load(fn2) # this is a numpy wrapper, for this we use [()]
    propagate3D(dataDict[()], inputDict)
