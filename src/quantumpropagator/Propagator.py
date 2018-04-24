import numpy as np
import quantumpropagator.EMPulse as pp

def rk4Ene3d(f, t, y, inp):
    '''
    Runge Kutta with 3d grid as y
    f -> function that returns the derivative value
    t -> time at which the derivative is calculated
    y -> the numpy array
    inp -> dictionaries with immutable objects
    '''
    h = inp['h']
    k1 = h * f(t, y, inp)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1, inp)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2, inp)
    k4 = h * f(t + h, y + k3, inp)
    return y + (k1 + k2 + k2 + k3 + k3 + k4) / 6

def derivative1dGam(t,GRID,inp):
    '''
    Propagator 1d on Gamma
    '''
    new = np.empty_like(GRID)
    return(new)

def derivative1dPhi(t,GRID,inp):
    '''
    ORA SI BALLA
    Propagator 1d to have some fun
    '''
    new = np.empty_like(GRID)
    for p in np.arange(inp['phiL']):
        G = GRID[p]
        V = inp['potCube'][p]
        K = inp['kinCube'][p]
        # derivatives in phi
        if p == 0:
            d2G_dp2 = (-GRID[p+2]+16*GRID[p+1]-30*GRID[p]) / (12 * inp['dphi']**2)

        elif p == 1:
            d2G_dp2 = (-GRID[p+2]+16*GRID[p+1]-30*GRID[p]+16*GRID[p-1]) / (12 * inp['dphi']**2)

        elif p == inp['phiL']-2:
            d2G_dp2 = (+16*GRID[p+1]-30*GRID[p]+16*GRID[p-1]-GRID[p-2]) / (12 * inp['dphi']**2)

        elif p == inp['phiL']-1:
            d2G_dp2 = (-30*GRID[p]+16*GRID[p-1]-GRID[p-2]) / (12 * inp['dphi']**2)

        else:
            d2G_dp2 = (-GRID[p+2]+16*GRID[p+1]-30*GRID[p]+16*GRID[p-1]-GRID[p-2]) / (12 * inp['dphi']**2)

        Tpp = K[0,2] * d2G_dp2
        Ttot = Tpp
        Vtot = V * G

        #print()
        #print('K:\n{}'.format(K))
        #print('G:\n{}'.format(G))
        #print('d2: {}'.format(d2G_dp2))
        #print('T: {}'.format(Tpp))
        #print('({})    Ttot: {}      Vtot: {}   elem: {}'.format(p,Ttot,Vtot, (-1j * (Ttot+Vtot))))

        new[p] = -1j * (Ttot+Vtot)
    return new

def derivative2dGamThe(t,GRID,inp):
    '''
    derivative done for a 2d Grid on the angles
    '''

    new = np.empty_like(GRID)
    kintotSum = 0
    pottotSum = 0
    for g in np.arange(inp['gamL']):
       for t in np.arange(inp['theL']):
           G = GRID[g,t]
           V = inp['potCube'][g,t,0]
           K = inp['kinCube'][g,t]

           # derivatives in gam
           if g == 0:
               dG_dg   = (GRID[g+1,t]) / (2 * inp['dgam'])
               d2G_dg2 = (-GRID[g+2,t]+16*GRID[g+1,t]-30*GRID[g,t]) / (12 * inp['dgam']**2)
               d2G_dgt_numerator_g = -GRID[g+1,t]

           elif g == 1:
               dG_dg   = (GRID[g+1,t]-GRID[g-1,t]) / (2 * inp['dgam'])
               d2G_dg2 = (-GRID[g+2,t]+16*GRID[g+1,t]-30*GRID[g,t]+16*GRID[g-1,t]) / (12 * inp['dgam']**2)
               d2G_dgt_numerator_g = -GRID[g+1,t] -GRID[g-1,t]

           elif g == inp['gamL']-2:
               dG_dg   = (GRID[g+1,t]-GRID[g-1,t]) / (2 * inp['dgam'])
               d2G_dg2 = (+16*GRID[g+1,t]-30*GRID[g,t]+16*GRID[g-1,t]-GRID[g-2,t]) / (12 * inp['dgam']**2)
               d2G_dgt_numerator_g = -GRID[g+1,t] -GRID[g-1,t]

           elif g == inp['gamL']-1:
               dG_dg   = (-GRID[g-1,t]) / (2 * inp['dgam'])
               d2G_dg2 = (-30*GRID[g,t]+16*GRID[g-1,t]-GRID[g-2,t]) / (12 * inp['dgam']**2)
               d2G_dgt_numerator_g = -GRID[g-1,t]

           else:
               dG_dg   = (GRID[g+1,t]-GRID[g-1,t]) / (2 * inp['dgam'])
               d2G_dg2 = (-GRID[g+2,t]+16*GRID[g+1,t]-30*GRID[g,t]+16*GRID[g-1,t]-GRID[g-2,t]) / (12 * inp['dgam']**2)
               d2G_dgt_numerator_g = -GRID[g+1,t] -GRID[g-1,t]

           # derivatives in the
           if t == 0:
               dG_dt   = (GRID[g,t+1]) / (2 * inp['dthe'])
               d2G_dt2 = (-GRID[g,t+2]+16*GRID[g,t+1]-30*GRID[g,t]) / (12 * inp['dthe']**2)
               d2G_dgt_numerator_t = -GRID[g,t+1]

           elif t == 1:
               dG_dt   = (GRID[g,t+1]-GRID[g,t-1]) / (2 * inp['dthe'])
               d2G_dt2 = (-GRID[g,t+2]+16*GRID[g,t+1]-30*GRID[g,t]+16*GRID[g,t-1]) / (12 * inp['dthe']**2)
               d2G_dgt_numerator_t = -GRID[g,t+1] -GRID[g,t-1]

           elif t == inp['theL']-2:
               dG_dt   = (GRID[g,t+1]-GRID[g,t-1]) / (2 * inp['dthe'])
               d2G_dt2 = (+16*GRID[g,t+1]-30*GRID[g,t]+16*GRID[g,t-1]-GRID[g,t-2]) / (12 * inp['dthe']**2)
               d2G_dgt_numerator_t = -GRID[g,t+1] -GRID[g,t-1]

           elif t == inp['theL']-1:
               dG_dt   = (-GRID[g,t-1]) / (2 * inp['dthe'])
               d2G_dt2 = (-30*GRID[g,t]+16*GRID[g,t-1]-GRID[g,t-2]) / (12 * inp['dthe']**2)
               d2G_dgt_numerator_t = -GRID[g,t-1]

           else:
               dG_dt   = (GRID[g,t+1]-GRID[g,t-1]) / (2 * inp['dthe'])
               d2G_dt2 = (-GRID[g,t+2]+16*GRID[g,t+1]-30*GRID[g,t]+16*GRID[g,t-1]-GRID[g,t-2]) / (12 * inp['dthe']**2)
               d2G_dgt_numerator_t = -GRID[g,t+1] -GRID[g,t-1]


           # cross terms: they're 2?
           if g == 0 or t == 0:
               d2G_dgt_numerator_cross_1 = 0
           else:
               d2G_dgt_numerator_cross_1 = +GRID[g-1,t-1]

           if g == inp['gamL']-1 or t == inp['theL']-1:
               d2G_dgt_numerator_cross_2 = 0
           else:
               d2G_dgt_numerator_cross_2 = +GRID[g+1,t+1]


           d2G_dgt_numerator = d2G_dgt_numerator_g + d2G_dgt_numerator_t + d2G_dgt_numerator_cross_1 + d2G_dgt_numerator_cross_2 + 2*G
           d2G_dgt = d2G_dgt_numerator/(2*inp['dgam']*inp['dthe'])
           d2G_dtg = d2G_dgt

           # T elements
           Tgg = K[4,0] * G + K[4,1] * dG_dg + K[4,2] * d2G_dg2
           Tgt = K[5,0] * G + K[5,1] * dG_dg + K[5,2] * d2G_dgt
           Ttg = K[7,0] * G + K[7,1] * dG_dt + K[7,2] * d2G_dtg
           Ttt = K[8,0] * G + K[8,1] * dG_dt + K[8,2] * d2G_dt2

           Ttot = Tgg + Tgt + Ttg + Ttt
           Vtot = V * G
           kintotSum += Ttot
           pottotSum += Vtot

           prr = False
           if prr == True:
               print()
               print(K)
               print('d1: {:e} {:e}'.format(dG_dg,dG_dt))
               print('d2: {:e} {:e}'.format(d2G_dg2,d2G_dt2))
               print('T: {:e} {:e} {:e} {:e}'.format(Tgg, Tgt, Ttg, Ttt))
               print('({},{})    Ttot: {:.2f}      Vtot: {:.2f}   elem: {:.2f}'.format(g,t,Ttot,Vtot, (-1j * (Ttot+Vtot))))

           new[g,t] = -1j * (Ttot+Vtot)
    #print('Sum on the grid -> Kin {:e} {:+e} i ###  Pot {:e} {:+e} i'.format(kintotSum.real,kintotSum.imag, pottotSum.real, pottotSum.imag))
    return new

def derivative3d(t,GRID,inp):
    '''
    ORA SI BALLA
    3d propagator using kinetic madness

    THE PULSE NEEDS TO BE INSIDE HERE BECAUSE IT DEPENDS ON T AND RK calls it 4 times

    '''

    npad = (2,2)
    GRID = np.pad(GRID, pad_width=npad, mode='constant', constant_values=0)
    new = np.empty_like(GRID)
    # I want to loop through all point that are 2 points far away from the border.
    # np.arange(10-4)+2 -> [2, 3, 4, 5, 6, 7]
    for p in np.arange(inp['phiL']-4)+2:
         for g in np.arange(inp['gamL']-4)+2:
            for t in np.arange(inp['theL']-4)+2:
                G = GRID[p,g,t]
                V = inp['potCube'][p-2,g-2,t-2,0] # when we use PAD everything is shifted
                K = inp['kinCube'][p-2,g-2,t-2] # when we use PAD everything is shifted
                # derivatives in phi
                dG_dp   = (GRID[p+1,g,t]-GRID[p-1,g,t]) / (2 * inp['dphi'])
                d2G_dp2 = (-GRID[p+2,g,t]+16*GRID[p+1,g,t]-30*GRID[p,g,t]+16*GRID[p-1,g,t]-GRID[p-2,g,t]) / (12 * inp['dphi']**2)

                # derivatives in gam
                dG_dg   = (GRID[p,g+1,t]-GRID[p,g-1,t]) / (2 * inp['dgam'])
                d2G_dg2 = (-GRID[p,g+2,t]+16*GRID[p,g+1,t]-30*GRID[p,g,t]+16*GRID[p,g-1,t]-GRID[p,g-2,t]) / (12 * inp['dgam']**2)

                # derivatives in the
                dG_dt   = (GRID[p,g,t+1]-GRID[p,g,t-1]) / (2 * inp['dthe'])
                d2G_dt2 = (-GRID[p,g,t+2]+16*GRID[p,g,t+1]-30*GRID[p,g,t]+16*GRID[p,g,t-1]-GRID[p,g,t-2]) / (12 * inp['dthe']**2)

                # cross terms: they're 6
                d2G_dpg = - (GRID[p+1,g,t]+GRID[p-1,g,t]+GRID[p,g+1,t]+GRID[p,g-1,t]-2*G-GRID[p+1,g+1,t]-GRID[p-1,g-1,t])/(2*inp['dphi']*inp['dgam'])
                d2G_dpt = - (GRID[p+1,g,t]+GRID[p-1,g,t]+GRID[p,g,t+1]+GRID[p,g,t-1]-2*G-GRID[p+1,g,t+1]-GRID[p-1,g,t-1])/(2*inp['dphi']*inp['dthe'])
                d2G_dgp = d2G_dpg
                d2G_dgt = - (GRID[p,g+1,t]+GRID[p,g-1,t]+GRID[p,g,t+1]+GRID[p,g,t-1]-2*G-GRID[p,g+1,t+1]-GRID[p,g-1,t-1])/(2*inp['dgam']*inp['dthe'])
                d2G_dtp = d2G_dpt
                d2G_dtg = d2G_dgt

                # T elements
                Tpp =                               K[0,2] * d2G_dp2
                Tpg =              K[1,1] * dG_dp + K[1,2] * d2G_dpg
                Tpt =              K[2,1] * dG_dp + K[2,2] * d2G_dpt
                Tgp =              K[3,1] * dG_dg + K[3,2] * d2G_dgp
                Tgg = K[4,0] * G + K[4,1] * dG_dg + K[4,2] * d2G_dg2
                Tgt = K[5,0] * G + K[5,1] * dG_dg + K[5,2] * d2G_dgt
                Ttp =              K[6,1] * dG_dt + K[6,2] * d2G_dtp
                Ttg = K[7,0] * G + K[7,1] * dG_dt + K[7,2] * d2G_dtg
                Ttt = K[8,0] * G + K[8,1] * dG_dt + K[8,2] * d2G_dt2

                Ttot = Tpp + Tpg + Tpt + Tgp + Tgg + Tgt + Ttp + Ttg + Ttt
                Vtot = V * G

                #print()
                #print(K)
                #print('d1: {} {} {}'.format(dG_dp,dG_dg,dG_dt))
                #print('d2: {} {} {}'.format(d2G_dp2,d2G_dg2,d2G_dt2))
                #print('T: {} {} {} {} {} {} {} {} {}'.format(Tpp, Tpg, Tpt, Tgp, Tgg, Tgt, Ttp, Ttg, Ttt))
                #print('({},{},{})    Ttot: {}      Vtot: {}   elem: {}'.format(p,g,t,Ttot,Vtot, (-1j * (Ttot+Vtot))))

                new[p,g,t] = -1j * (Ttot+Vtot)
    return new[2:-2, 2:-2, 2:-2]


def rk4Ene1dSLOW(f, t, y, h, pulse, ene, dipo, NAC, Gele, nstates,gridN,kaxisR,reducedMass,absorbPot):
    k1 = h * f(t, y, pulse[0], ene, dipo, NAC, Gele, nstates,gridN,kaxisR,reducedMass,absorbPot)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1, pulse[1], ene, dipo, NAC, Gele, nstates,gridN,kaxisR,reducedMass,absorbPot)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2, pulse[1], ene, dipo, NAC, Gele, nstates,gridN,kaxisR,reducedMass,absorbPot)
    k4 = h * f(t + h, y + k3, pulse[2], ene, dipo, NAC, Gele, nstates,gridN,kaxisR,reducedMass,absorbPot)
    return y + (k1 + k2 + k2 + k3 + k3 + k4) / 6

def derivative1d(t, GRID, pulseV, matVG, matMuG, matNACG, matGELEG, nstates, gridN,
                 kaxisR, reducedMass, absP):
    '''
    This is the correct one... holy cow holy cow

            after summa[Ici] = con * hamilt
            #if g > 971 and g < 1000 and summa[Ici] > 0.00001:
            #print('Summa -> {} -> {}'.format(Ici,summa[Ici]))
    '''
    con = -1j
    new = np.empty((nstates, gridN), dtype=complex)
    (doublederivative,singlederivative) = NuclearKinetic1d(GRID, kaxisR, nstates, gridN)
    for g in range(gridN):
        states  = GRID[:,g]
        d2R     = doublederivative[:,g]
        dR      = singlederivative[:,g]
        summa   = np.zeros(nstates, dtype = complex)
        for Ici in range(nstates):
            hamilt = sum(HamiltonianEle1d(Ici, Icj, matVG[g], matMuG[g], pulseV, d2R, dR, g,
                         states[Icj], reducedMass, matNACG[g], matGELEG[g], absP[g])
                         for Icj in range(nstates))
            summa[Ici] = con * hamilt
        new[:,g] = summa
    return new

def HamiltonianEle1d(Ici,Icj,matV,matMu,pulseV,d2R,dR,g,cj,reducedMass,tau,gMat,absP):
    '''
    WATCH OUT THIS [2] IS TO MAKE THE PULSE 1D IMPORTANT

    matMu[0, Ici, Icj]  <- the 0 here should be replaced
    iEMuj = np.dot(pulseV[2],muij)

    super important PULSE AND NAC WILL SOON BE VECTORS
    '''

    if Ici == Icj:
        iTnj = d2R[Icj]/(2*reducedMass)    # this is already negative because of the FT
        iH0j = matV[Ici] * cj
        absP = - 1j * absP * cj
    else:
        iTnj = 0.0
        iH0j = 0.0
        absP = 0.0
    muij    = matMu[0, Ici, Icj]
    iEMuj   = - (pulseV[2]*muij) * cj
    nac1    = - (tau[Ici, Icj] * dR[Icj]) / reducedMass
    sumTaus = np.dot(tau[Ici,:],tau[:,Icj])
    nac2    = - ((sumTaus + gMat[Ici, Icj]) / (2*reducedMass)) * cj

    #'''Print every possible output'''
       #string = '{}({},{}) Tn={} H0={} muij={} pul={} EMu={} tij={} nij={} st={} gij={}'.format(
       #                g,    Ici,Icj,    iTnj,  iH0j,   muij, pulseV[2], iEMuj, tau[Ici, Icj], inacj, sumTaus, matG[Ici, Icj])
    #string = '{}({},{}) sinDj={} tij={} gij={} nac1={} nac2={}'.format(
    #           g,Ici,Icj, dR[Icj], tau[Ici, Icj], gMat[Ici, Icj], nac1, nac2)
    #print(string)

    return iH0j + iEMuj + iTnj + absP + nac1 + nac2

def NuclearKinetic1d(states,kaxisR,nstates,gridN):
    ''' watch this function for "in place" operations '''
    dou    = np.empty((nstates,gridN),dtype = complex)
    sin    = np.empty((nstates,gridN),dtype = complex)
    for i in range(nstates):
        transform  = np.fft.fft(states[i])
        doubleDeri = np.apply_along_axis(lambda t: doubleDeronK(*t), 0, np.stack((transform,kaxisR)))
        dou[i] = np.fft.ifft(doubleDeri)
        singleDeri = np.apply_along_axis(lambda t: singleDerOnK(*t), 0, np.stack((transform,kaxisR)))
        sin[i] = np.fft.ifft(singleDeri)
    return (dou,sin)

def doubleDeronK(a,b):
    return a*(b**2)

def singleDerOnK(a,b):
    return 1j*a*b

def createXaxisReciprocalspace1d(gridN,deltaX):
    rolln   = divideByTwo(gridN)
    limit   = np.pi/deltaX
    kaxis   = np.linspace(-limit,limit,gridN)
    return np.roll(kaxis, rolln)

def divideByTwo(n):
    '''
    From integer to the correct number to roll Xaxis in FT
    '''
    if n % 2 == 0:
        a=int(n/2)
    else:
        a=int(n/2+1)
    return a

#def calculateKinetic(t, GRID, pulseV, matVG, matMuG, nstates,gridN,kaxisR,reducedMass):
#    doublederivative = NuclearKinetic1d(GRID, kaxisR, nstates, gridN, reducedMass)
#    kinetic  = np.vdot(GRID,doublederivative)
#    print('Kinetic Energy -> ', np.real(kinetic))
#    return kinetic

def calculateTotal(t, GRID, pulseV, matVG, matMuG, matNACG, matGELEG, nstates, gridN,
                   kaxisR, reducedMass, absP):
    new = np.empty((nstates,gridN),dtype = complex)
    (doublederivative,singlederivative) = NuclearKinetic1d(GRID, kaxisR, nstates, gridN)
    for g in range(gridN):
        states = GRID[:,g]
        d2R = doublederivative[:,g]
        dR = singlederivative[:,g]
        summa = np.zeros(nstates, dtype = complex)
        for Ici in range(nstates):
            hamilt = sum(HamiltonianEle1d(Ici, Icj, matVG[g], matMuG[g], pulseV[0], d2R, dR, g,
                         states[Icj], reducedMass, matNACG[g], matGELEG[g], absP[g])
                         for Icj in range(nstates))
            summa[Ici] = hamilt
        new[:,g] = summa
    total = np.vdot(GRID,new)
#    print('Total Energy   -> ', np.real(total))
    return total

def rk6Ene1dSLOW(f, t, y, h, pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass):
    k1 = h * f( t            , y , pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    k2 = h * f( t + h/3      , y + k1/3 , pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    k3 = h * f( t + 2 * h / 3, y + 2 * k2 / 3 , pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    k4 = h * f( t + h / 3    , y + ( k1 + 4 * k2 - k3 ) / 12 , pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    k5 = h * f( t + h / 2    , y + (-k1 + 18 * k2 - 3 * k3 -6 * k4)/16 , pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    k6 = h * f( t + h / 2    , y + (9 * k2 - 3 * k3 - 6 * k4 + 4 * k5)/8 , pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    k7 = h * f( t + h        , y + (9 * k1 - 36 * k2 + 63 * k3 + 72 * k4 - 64 * k5)/44 , pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    return y + (11 * k1 + 81 * k3 + 81 * k4 - 32 * k5 - 32 * k6 + 11 * k7) / 120

def EULER1d(f, t, y, h, pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass):
    k1 = h * f(t, y, pulse, ene, dipo, nstates,gridN,kaxisR,reducedMass)
    return y + k1


###########################
# Single Point integrator #
###########################


def rk4Ene(f, t, y, h, pulse, matV, matMu):
    '''
    Runge-Kutta integrator for VECTORS, 'cause y here is gonna be an array of coefficients
    f = the derivativeC function :: Double,[Complex],[[Double]],[3[[Double]]] -> [Complex]
    t = time :: Double
    y = coefficients vector :: [Complex]
    pulse = list of pulse arguments
    matV = energies diagonal matrix :: [[Double]]
    matMu = transition dipole matrix :: [3[[Double]]]
    '''
    k1 = h * f(t, y, pulse, matV, matMu)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1, pulse, matV, matMu)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2, pulse, matV, matMu)
    k4 = h * f(t + h, y + k3, pulse, matV, matMu)
    ny = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return ny


def derivativeC(t,states,pulse,matV,matMu):
    '''
    The derivative for our coefficients. It will apply the hamiltonian to every ci coefficient.
    $\color{violet}\dot{c_i}=\dfrac{\partial c_i(t)}{\partial t} = \dfrac{1}{i\hbar} \sum_j \hat{H}_{ij} c_j(t)$
    '''
    con     = -1j
    pulseV  = pp.userPulse(t,pulse)
    nstates = states.size
    summa   = np.zeros(nstates, dtype = complex)
    for Ici in range(nstates):
       #hamilt = np.sum([ HamiltonianEle(Ici,Icj,matV,matMu,pulseV) * states[Icj] for Icj in range(nstates) ])
       hamilt = sum(HamiltonianEle(Ici,Icj,matV,matMu,pulseV) * states[Icj] for Icj in range(nstates))
       summa[Ici]= con*hamilt
    return summa


def HamiltonianEle(Ici,Icj,matV,matMu,pulseV):
    '''
    This is the hamiltonian element for a system that is frozen in one geometry (no kinetic energy).
    $\color{violet}H_{ij} = \langle i|\hat{H}^0|j\rangle \delta_{ij} + \hat{T}_N - \vec{E}_0(t) \vec{\mu}_{ij}$
    The pulse depends on time
    pulsev = the particular value of the external pulse at time t :: [x,y,z]
    muij = the element of the transition dipole matrix :: [x,y,z]
    the line "muij = matMu[:3, Ici, Icj]" creates a 3 long vector with [1,2,3][Ici][Icj]
    '''
    iH0j = matV[Ici, Icj]
    muij = matMu[:3, Ici, Icj]
    secondterm = np.dot(pulseV,muij)
    return iH0j - secondterm

#def derivative2dGamThe(t,GRID,inp):
#    '''
#    derivative done for a 2d Grid on the angles
#
#    THIS IS THE PAD VERSION AND IT IS ACTUALLY A LITTLE WRONG...
#    '''
#
#    npad = (2,2)
#    GRID = np.pad(GRID, pad_width=npad, mode='constant', constant_values=0)
#    new = np.empty_like(GRID)
#    # I want to loop through all point that are 2 points far away from the border.
#    # np.arange(10-4)+2 -> [2, 3, 4, 5, 6, 7]
#    for g in np.arange(inp['gamL']-4)+2:
#       for t in np.arange(inp['theL']-4)+2:
#           G = GRID[g,t]
#           V = inp['potCube'][g-2,t-2,0] # when we use PAD everything is shifted
#           K = inp['kinCube'][g-2,t-2] # when we use PAD everything is shifted
#
#           # derivatives in gam
#           dG_dg   = (GRID[g+1,t]-GRID[g-1,t]) / (2 * inp['dgam'])
#           d2G_dg2 = (-GRID[g+2,t]+16*GRID[g+1,t]-30*GRID[g,t]+16*GRID[g-1,t]-GRID[g-2,t]) / (12 * inp['dgam']**2)
#
#           # derivatives in the
#           dG_dt   = (GRID[g,t+1]-GRID[g,t-1]) / (2 * inp['dthe'])
#           d2G_dt2 = (-GRID[g,t+2]+16*GRID[g,t+1]-30*GRID[g,t]+16*GRID[g,t-1]-GRID[g,t-2]) / (12 * inp['dthe']**2)
#
#           # cross terms: they're 2?
#           d2G_dgt = - (GRID[g+1,t]+GRID[g-1,t]+GRID[g,t+1]+GRID[g,t-1]-2*G-GRID[g+1,t+1]-GRID[g-1,t-1])/(2*inp['dgam']*inp['dthe'])
#           d2G_dtg = d2G_dgt
#
#           # T elements
#           Tgg = K[4,0] * G + K[4,1] * dG_dg + K[4,2] * d2G_dg2
#           Tgt = K[5,0] * G + K[5,1] * dG_dg + K[5,2] * d2G_dgt
#           Ttg = K[7,0] * G + K[7,1] * dG_dt + K[7,2] * d2G_dtg
#           Ttt = K[8,0] * G + K[8,1] * dG_dt + K[8,2] * d2G_dt2
#
#           Ttot = Tgg + Tgt + Ttg + Ttt
#           Vtot = V * G
#
#           prr = False
#           if prr == True:
#               print()
#               print(K)
#               print('d1: {} {}'.format(dG_dg,dG_dt))
#               print('d2: {} {}'.format(d2G_dg2,d2G_dt2))
#               print('T: {} {} {} {}'.format(Tgg, Tgt, Ttg, Ttt))
#               print('({},{})    Ttot: {}      Vtot: {}   elem: {}'.format(g,t,Ttot,Vtot, (-1j * (Ttot+Vtot))))
#
#           new[g,t] = -1j * (Ttot+Vtot)
#    return new[2:-2, 2:-2]
#
