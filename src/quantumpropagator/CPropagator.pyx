# distutils: extra_compile_args = -fopenmp
# distutils: extra_link_args = -fopenmp

import numpy as np
cimport numpy as np
#import quantumpropagator.EMPulse as pp
cimport cython
from cython.parallel import prange
from libc.stdio cimport printf

#import pyximport; pyximport.install()

# This is needed for complex numbers
cdef extern from "complex.h":
        double complex cexp(double complex)

def Crk4Ene3d(f, t, y, inp):
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

def pulZe(t, param_Pulse):
    Ed,omega,sigma,phase,t0 = param_Pulse
    num = (t-t0)**2
    den = 2*(sigma**2)
    if (den == 0):
        result = 0.0
    else:
        result = Ed * (np.cos(omega*t+phase)) * np.exp(-num/den)
    return result

def CextractEnergy3dMu(t,GRID,inp):
    '''wrapper for 3d integrator in Kinetic-Potential mode'''
    #print('ENERGY -> t: {} , wf inside: {}'.format(t,GRID.shape))
    return np.asarray(Cderivative3dMu_cyt(t,GRID,inp,0))

def Cderivative3dMu(t,GRID,inp):
    '''wrapper for 3d integrator'''
    #print('PROPAG -> t: {} , wf inside: {}'.format(t,GRID.shape))
    return np.asarray(Cderivative3dMu_cyt(t,GRID,inp,1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Cderivative3dMu_cyt(double time, double complex [:,:,:,:] GRID, dict inp, int selector):
    '''
    derivative done for a 3d Grid on all the coordinates
    t :: Double -> time
    GRID :: np.array[:,:,:,:] <- wavefunction[phis,gams,thes,states]
    inp :: dictionary of various inputs
    export OMP_NUM_THREADS=10
    '''
    cdef:
        int s,p,g,t,phiL=inp['phiL'],gamL=inp['gamL'],theL=inp['theL'],nstates=inp['nstates']
        int d,carte
        double dphi=inp['dphi'],dgam=inp['dgam'],dthe=inp['dthe'],V
        double [:,:,:,:] Vm = inp['potCube']
        double [:,:,:,:,:] Km = inp['kinCube']
        double [:,:,:,:,:,:] Dm = inp['dipCube']
        double [:,:,:,:,:,:] Nm = inp['nacCube']
        double [:] pulseV
        double complex [:,:,:,:] new, kinS, potS
        double complex I = -1j
        double complex dG_dp, d2G_dp2, dG_dg, d2G_dg2, dG_dt, d2G_dt2, G
        double complex dG_dp_oth, dG_dg_oth, dG_dt_oth
        double complex d2G_dcross_numerator_p,d2G_dcross_numerator_g,d2G_dcross_numerator_t
        double complex d2G_dpg_numerator_cross_1,d2G_dpg_numerator_cross_2,d2G_dpg_numerator
        double complex d2G_dpt_numerator_cross_1,d2G_dpt_numerator_cross_2,d2G_dpt_numerator
        double complex d2G_dgt_numerator_cross_1,d2G_dgt_numerator_cross_2,d2G_dgt_numerator
        double complex d2G_dpg,d2G_dpt,d2G_dgt,d2G_dgp,d2G_dtp,d2G_dtg
        double complex Tpp,Tpg,Tpt,Tgp,Tgg,Tgt,Ttp,Ttg,Ttt
        double complex Ttot,Vtot,Mtot,Ntot

    new = np.empty_like(GRID)
    kinS = np.empty_like(GRID)
    potS = np.empty_like(GRID)

    pulseV = np.empty((3))

    pulseV[0] = pulZe(time,inp['pulseX'])
    pulseV[1] = pulZe(time,inp['pulseY'])
    pulseV[2] = pulZe(time,inp['pulseZ'])

    #for s in range(nstates):
    for s in prange(nstates, nogil=True):   # first state loop.
        for p in range(phiL):
            for g in range(gamL):
               for t in range(theL):
                   G = GRID[p,g,t,s]
                   V = Vm[p,g,t,s]

                   # derivatives in phi
                   if p == 0:
                       #dG_dp   = (GRID[p+1,g,t,s]) / (2 * dphi)
                       dG_dp   = ((2/3)*GRID[p+1,g,t,s]+(-1/12)*GRID[p+2,g,t,s]) / dphi
                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s]

                   elif p == 1:
                       #dG_dp   = (GRID[p+1,g,t,s]-GRID[p-1,g,t,s]) / (2 * dphi)
                       dG_dp   = ((-2/3)*GRID[p-1,g,t,s]+(2/3)*GRID[p+1,g,t,s]+(-1/12)*GRID[p+2,g,t,s]) / dphi
                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]

                   elif p == phiL-2:
                       #dG_dp   = (GRID[p+1,g,t,s]-GRID[p-1,g,t,s]) / (2 * dphi)
                       dG_dp   = ((1/12)*GRID[p-2,g,t,s]+(-2/3)*GRID[p-1,g,t,s]+(2/3)*GRID[p+1,g,t,s]) / dphi
                       d2G_dp2 = (+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]

                   elif p == phiL-1:
                       #dG_dp   = (-GRID[p-1,g,t,s]) / (2 * dphi)
                       dG_dp   = ((1/12)*GRID[p-2,g,t,s]+(-2/3)*GRID[p-1,g,t,s]) / dphi
                       d2G_dp2 = (-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p-1,g,t,s]

                   else:
                       #dG_dp   = (GRID[p+1,g,t,s]-GRID[p-1,g,t,s]) / (2 * dphi)
                       dG_dp   = ((1/12)*GRID[p-2,g,t,s]+(-2/3)*GRID[p-1,g,t,s]+(2/3)*GRID[p+1,g,t,s]+(-1/12)*GRID[p+2,g,t,s]) / dphi
                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]

                   # derivatives in gam
                   if g == 0:
                       #dG_dg   = (GRID[p,g+1,t,s]) / (2 * dgam)
                       dG_dg   = ((2/3)*GRID[p,g+1,t,s]+(-1/12)*GRID[p,g+2,t,s]) / dgam
                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s]

                   elif g == 1:
                       #dG_dg   = (GRID[p,g+1,t,s]-GRID[p,g-1,t,s]) / (2 * dgam)
                       dG_dg   = ((-2/3)*GRID[p,g-1,t,s]+(2/3)*GRID[p,g+1,t,s]+(-1/12)*GRID[p,g+2,t,s]) / dgam
                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]

                   elif g == gamL-2:
                       #dG_dg   = (GRID[p,g+1,t,s]-GRID[p,g-1,t,s]) / (2 * dgam)
                       dG_dg   = ((1/12)*GRID[p,g-2,t,s]+(-2/3)*GRID[p,g-1,t,s]+(2/3)*GRID[p,g+1,t,s]) / dgam
                       d2G_dg2 = (+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]

                   elif g == gamL-1:
                       #dG_dg   = (-GRID[p,g-1,t,s]) / (2 * dgam)
                       dG_dg   = ((1/12)*GRID[p,g-2,t,s]+(-2/3)*GRID[p,g-1,t,s]) / dgam
                       d2G_dg2 = (-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g-1,t,s]

                   else:
                       #dG_dg   = (GRID[p,g+1,t,s]-GRID[p,g-1,t,s]) / (2 * dgam)
                       dG_dg   = ((1/12)*GRID[p,g-2,t,s]+(-2/3)*GRID[p,g-1,t,s]+(2/3)*GRID[p,g+1,t,s]+(-1/12)*GRID[p,g+2,t,s]) / dgam
                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]

                   # derivatives in the
                   if t == 0:
                       #dG_dt   = (GRID[p,g,t+1,s]) / (2 * dthe)
                       dG_dt   = ((2/3)*GRID[p,g,t+1,s]+(-1/12)*GRID[p,g,t+2,s]) / dthe
                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s]

                   elif t == 1:
                       #dG_dt   = (GRID[p,g,t+1,s]-GRID[p,g,t-1,s]) / (2 * dthe)
                       dG_dt   = ((-2/3)*GRID[p,g,t-1,s]+(2/3)*GRID[p,g,t+1,s]+(-1/12)*GRID[p,g,t+2,s]) / dthe
                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]

                   elif t == theL-2:
                       #dG_dt   = (GRID[p,g,t+1,s]-GRID[p,g,t-1,s]) / (2 * dthe)
                       dG_dt   = ((1/12)*GRID[p,g,t-2,s]+(-2/3)*GRID[p,g,t-1,s]+(2/3)*GRID[p,g,t+1,s]) / dthe
                       d2G_dt2 = (+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]

                   elif t == theL-1:
                       #dG_dt   = (-GRID[p,g,t-1,s]) / (2 * dthe)
                       dG_dt   = ((1/12)*GRID[p,g,t-2,s]+(-2/3)*GRID[p,g,t-1,s]) / dthe
                       d2G_dt2 = (-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t-1,s]

                   else:
                       #dG_dt   = (GRID[p,g,t+1,s]-GRID[p,g,t-1,s]) / (2 * dthe)
                       dG_dt   = ((1/12)*GRID[p,g,t-2,s]+(-2/3)*GRID[p,g,t-1,s]+(2/3)*GRID[p,g,t+1,s]+(-1/12)*GRID[p,g,t+2,s]) / dthe
                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]


                   # cross terms: they're thousands... 
                   if p == 0 or g == 0:
                       d2G_dpg_numerator_cross_1 = 0
                   else:
                       d2G_dpg_numerator_cross_1 = +GRID[p-1,g-1,t,s]
                   if p == phiL-1 or g == gamL-1:
                       d2G_dpg_numerator_cross_2 = 0
                   else:
                       d2G_dpg_numerator_cross_2 = +GRID[p+1,g+1,t,s]

                   if p == 0 or t == 0:
                       d2G_dpt_numerator_cross_1 = 0
                   else:
                       d2G_dpt_numerator_cross_1 = +GRID[p-1,g,t-1,s]
                   if p == phiL-1 or t == theL-1:
                       d2G_dpt_numerator_cross_2 = 0
                   else:
                       d2G_dpt_numerator_cross_2 = +GRID[p+1,g,t+1,s]

                   if g == 0 or t == 0:
                       d2G_dgt_numerator_cross_1 = 0
                   else:
                       d2G_dgt_numerator_cross_1 = +GRID[p,g-1,t-1,s]
                   if g == gamL-1 or t == theL-1:
                       d2G_dgt_numerator_cross_2 = 0
                   else:
                       d2G_dgt_numerator_cross_2 = +GRID[p,g+1,t+1,s]

                   # triple 0 or triplelast, we DO NOT NEED these term, as ANY of my terms in the kinetic energy depends on displacements along all three 
                   # coordinates... thus, no special cases where (p ==o or g == 0 or t == 0)

                   d2G_dpg_numerator = d2G_dcross_numerator_p + d2G_dcross_numerator_g + d2G_dpg_numerator_cross_1 + d2G_dpg_numerator_cross_2 + 2*G
                   d2G_dpt_numerator = d2G_dcross_numerator_p + d2G_dcross_numerator_t + d2G_dpt_numerator_cross_1 + d2G_dpt_numerator_cross_2 + 2*G
                   d2G_dgt_numerator = d2G_dcross_numerator_g + d2G_dcross_numerator_t + d2G_dgt_numerator_cross_1 + d2G_dgt_numerator_cross_2 + 2*G

                   d2G_dpg = d2G_dpg_numerator/(2*dphi*dgam)
                   d2G_dpt = d2G_dpt_numerator/(2*dphi*dthe)
                   d2G_dgt = d2G_dgt_numerator/(2*dgam*dthe)
                   d2G_dgp = d2G_dpg
                   d2G_dtp = d2G_dpt
                   d2G_dtg = d2G_dgt

                   # T elements (9)
                   Tpp = Km[p,g,t,0,0] * G + Km[p,g,t,0,1] * dG_dp + Km[p,g,t,0,2] * d2G_dp2
                   Tpg = Km[p,g,t,1,0] * G + Km[p,g,t,1,1] * dG_dp + Km[p,g,t,1,2] * d2G_dpg
                   Tpt = Km[p,g,t,2,0] * G + Km[p,g,t,2,1] * dG_dp + Km[p,g,t,1,2] * d2G_dpt

                   Tgp = Km[p,g,t,3,0] * G + Km[p,g,t,3,1] * dG_dg + Km[p,g,t,3,2] * d2G_dgp
                   Tgg = Km[p,g,t,4,0] * G + Km[p,g,t,4,1] * dG_dg + Km[p,g,t,4,2] * d2G_dg2
                   Tgt = Km[p,g,t,5,0] * G + Km[p,g,t,5,1] * dG_dg + Km[p,g,t,5,2] * d2G_dgt

                   Ttp = Km[p,g,t,6,0] * G + Km[p,g,t,6,1] * dG_dt + Km[p,g,t,6,2] * d2G_dtp
                   Ttg = Km[p,g,t,7,0] * G + Km[p,g,t,7,1] * dG_dt + Km[p,g,t,7,2] * d2G_dtg
                   Ttt = Km[p,g,t,8,0] * G + Km[p,g,t,8,1] * dG_dt + Km[p,g,t,8,2] * d2G_dt2

                   Ttot = (Tpp + Tpg + Tpt + Tgp + Tgg + Tgt + Ttp + Ttg + Ttt)
                   Vtot = V * G

                   # loop and sum on other states.
                   Mtot = 0
                   Ntot = 0

                   # this is state S with all the others. Summing up.

                   for d in range(nstates): # state s is where the outer loop is, d is where the inner loop is.
                       for carte in range(3): # carte is 'cartesian', meaning 0,1,2 -> x,y,z
                           # Mtot += -(pulseV[carte] * D[carte,s,d] ) * GRID[p,g,t,d]
                           # parallel version DOES NOT want += (better, it does not consider += as
                           # the serial version would. += works on shared reduction
                           # variables inside of the prange() loop
                           Mtot = Mtot - ((pulseV[carte] * Dm[p,g,t,carte,s,d] ) * GRID[p,g,t,d])

                           # NAC calculation

                           if p == 0:
                               dG_dp_oth = ((2/3)*GRID[p+1,g,t,d]+(-1/12)*GRID[p+2,g,t,d]) / dphi
                           elif p == 1:
                               dG_dp_oth = ((-2/3)*GRID[p-1,g,t,d]+(2/3)*GRID[p+1,g,t,d]+(-1/12)*GRID[p+2,g,t,d]) / dphi
                           elif p == phiL-2:
                               dG_dp_oth = ((1/12)*GRID[p-2,g,t,d]+(-2/3)*GRID[p-1,g,t,d]+(2/3)*GRID[p+1,g,t,d]) / dphi
                           elif p == phiL-1:
                               dG_dp_oth = ((1/12)*GRID[p-2,g,t,d]+(-2/3)*GRID[p-1,g,t,d]) / dphi
                           else:
                               dG_dp_oth = ((1/12)*GRID[p-2,g,t,d]+(-2/3)*GRID[p-1,g,t,d]+(2/3)*GRID[p+1,g,t,d]+(-1/12)*GRID[p+2,g,t,d]) / dphi

                           if g == 0:
                               dG_dg_oth = ((2/3)*GRID[p,g+1,t,d]+(-1/12)*GRID[p,g+2,t,d]) / dgam
                           elif g == 1:
                               dG_dg_oth = ((-2/3)*GRID[p,g-1,t,d]+(2/3)*GRID[p,g+1,t,d]+(-1/12)*GRID[p,g+2,t,d]) / dgam
                           elif g == gamL-2:
                               dG_dg_oth = ((1/12)*GRID[p,g-2,t,d]+(-2/3)*GRID[p,g-1,t,d]+(2/3)*GRID[p,g+1,t,d]) / dgam
                           elif g == gamL-1:
                               dG_dg_oth = ((1/12)*GRID[p,g-2,t,d]+(-2/3)*GRID[p,g-1,t,d]) / dgam
                           else:
                               dG_dg_oth = ((1/12)*GRID[p,g-2,t,d]+(-2/3)*GRID[p,g-1,t,d]+(2/3)*GRID[p,g+1,t,d]+(-1/12)*GRID[p,g+2,t,d]) / dgam

                           if t == 0:
                               dG_dt_oth = ((2/3)*GRID[p,g,t+1,d]+(-1/12)*GRID[p,g,t+2,d]) / dthe
                           elif t == 1:
                               dG_dt_oth = ((-2/3)*GRID[p,g,t-1,d]+(2/3)*GRID[p,g,t+1,d]+(-1/12)*GRID[p,g,t+2,d]) / dthe
                           elif t == theL-2:
                               dG_dt_oth = ((1/12)*GRID[p,g,t-2,d]+(-2/3)*GRID[p,g,t-1,d]+(2/3)*GRID[p,g,t+1,d]) / dthe
                           elif t == theL-1:
                               dG_dt_oth = ((1/12)*GRID[p,g,t-2,d]+(-2/3)*GRID[p,g,t-1,d]) / dthe
                           else:
                               dG_dt_oth = ((1/12)*GRID[p,g,t-2,d]+(-2/3)*GRID[p,g,t-1,d]+(2/3)*GRID[p,g,t+1,d]+(-1/12)*GRID[p,g,t+2,d]) / dthe


                           Ntot = Ntot - ( Nm[p,g,t,s,d,0] * dG_dp_oth + Nm[p,g,t,s,d,1] * dG_dg_oth + Nm[p,g,t,s,d,2] * dG_dt_oth)

                   if selector == 1:
                       # I = -i
                       new[p,g,t,s] = I * (Ttot+Vtot+Mtot+Ntot)
                       #new[p,g,t,s] = I * (Ttot+Vtot+Mtot)
                   else:
                       kinS[p,g,t,s] = Ttot
                       potS[p,g,t,s] = Vtot
                       #lol = Tpp.real * Tpp.real + Tpp.imag * Tpp.imag
                       #if lol > 0.00001:
                       #    printf("%i %i %i %i %f\n", p,g,t,s,lol)
    if selector == 1:
        return(new)
    else:
        return(kinS,potS)


#########################
# 2D code starting here #
#########################

# GAM THE

def Cenergy_2d_GamThe(t,GRID,inp):
    '''wrapper'''
    return np.asarray(Cderivative2D_GamThe_Mu(t,GRID,inp,0))

def Cderivative_2d_GamThe(t,GRID,inp):
    '''wrapper'''
    return np.asarray(Cderivative2D_GamThe_Mu(t,GRID,inp,1))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Cderivative2D_GamThe_Mu(double time,double complex [:,:,:] GRID, dict inp, int selector):
    '''
    derivative done for a 2d Grid on the angles
    '''

    cdef:
        int g,t,gamL=inp['gamL'],theL=inp['theL'],nstates=inp['nstates']
        double dgam = inp['dgam'], dthe = inp['dthe'], V
        double complex dG_dg, d2G_dg2, d2G_dgt_numerator_g, d2G_dgt_numerator_t, dG_dt, d2G_dt2
        double complex d2G_dgt_numerator_cross_1, d2G_dgt_numerator_cross_2, d2G_dgt_numerator, G
        double complex d2G_dgt, d2G_dtg, Tgg,Tgt,Ttg,Ttt,Ttot,Vtot,Mtot
        double [:,:,:] Vm = inp['potCube']
        double [:,:,:,:] Km = inp['kinCube']
        double [:,:,:,:,:] Dm = inp['dipCube']
        double [:] pulseV
        double complex [:,:,:] new, kinS, potS
        double complex I = -1j

    new = np.empty_like(GRID)
    kinS = np.empty_like(GRID)
    potS = np.empty_like(GRID)

    pulseV = np.empty((3))

    pulseV[0] = pulZe(time,inp['pulseX'])
    pulseV[1] = pulZe(time,inp['pulseY'])
    pulseV[2] = pulZe(time,inp['pulseZ'])

    for s in range(nstates):
        for g in range(gamL):
           for t in range(theL):
               G = GRID[g,t,s]
               V = Vm[g,t,s]

               # derivatives in gam
               if g == 0:
                   dG_dg   = (GRID[g+1,t,s]) / (2 * dgam)
                   d2G_dg2 = (-GRID[g+2,t,s]+16*GRID[g+1,t,s]-30*GRID[g,t,s]) / (12 * dgam**2)
                   d2G_dgt_numerator_g = -GRID[g+1,t,s]

               elif g == 1:
                   dG_dg   = (GRID[g+1,t,s]-GRID[g-1,t,s]) / (2 * dgam)
                   d2G_dg2 = (-GRID[g+2,t,s]+16*GRID[g+1,t,s]-30*GRID[g,t,s]+16*GRID[g-1,t,s]) / (12 * dgam**2)
                   d2G_dgt_numerator_g = -GRID[g+1,t,s] -GRID[g-1,t,s]

               elif g == gamL-2:
                   dG_dg   = (GRID[g+1,t,s]-GRID[g-1,t,s]) / (2 * dgam)
                   d2G_dg2 = (+16*GRID[g+1,t,s]-30*GRID[g,t,s]+16*GRID[g-1,t,s]-GRID[g-2,t,s]) / (12 * dgam**2)
                   d2G_dgt_numerator_g = -GRID[g+1,t,s] -GRID[g-1,t,s]

               elif g == gamL-1:
                   dG_dg   = (-GRID[g-1,t,s]) / (2 * dgam)
                   d2G_dg2 = (-30*GRID[g,t,s]+16*GRID[g-1,t,s]-GRID[g-2,t,s]) / (12 * dgam**2)
                   d2G_dgt_numerator_g = -GRID[g-1,t,s]

               else:
                   dG_dg   = (GRID[g+1,t,s]-GRID[g-1,t,s]) / (2 * dgam)
                   d2G_dg2 = (-GRID[g+2,t,s]+16*GRID[g+1,t,s]-30*GRID[g,t,s]+16*GRID[g-1,t,s]-GRID[g-2,t,s]) / (12 * dgam**2)
                   d2G_dgt_numerator_g = -GRID[g+1,t,s] -GRID[g-1,t,s]

               # derivatives in the
               if t == 0:
                   dG_dt   = (GRID[g,t+1,s]) / (2 * dthe)
                   d2G_dt2 = (-GRID[g,t+2,s]+16*GRID[g,t+1,s]-30*GRID[g,t,s]) / (12 * dthe**2)
                   d2G_dgt_numerator_t = -GRID[g,t+1,s]

               elif t == 1:
                   dG_dt   = (GRID[g,t+1,s]-GRID[g,t-1,s]) / (2 * dthe)
                   d2G_dt2 = (-GRID[g,t+2,s]+16*GRID[g,t+1,s]-30*GRID[g,t,s]+16*GRID[g,t-1,s]) / (12 * dthe**2)
                   d2G_dgt_numerator_t = -GRID[g,t+1,s] -GRID[g,t-1,s]

               elif t == theL-2:
                   dG_dt   = (GRID[g,t+1,s]-GRID[g,t-1,s]) / (2 * dthe)
                   d2G_dt2 = (+16*GRID[g,t+1,s]-30*GRID[g,t,s]+16*GRID[g,t-1,s]-GRID[g,t-2,s]) / (12 * dthe**2)
                   d2G_dgt_numerator_t = -GRID[g,t+1,s] -GRID[g,t-1,s]

               elif t == theL-1:
                   dG_dt   = (-GRID[g,t-1,s]) / (2 * dthe)
                   d2G_dt2 = (-30*GRID[g,t,s]+16*GRID[g,t-1,s]-GRID[g,t-2,s]) / (12 * dthe**2)
                   d2G_dgt_numerator_t = -GRID[g,t-1,s]

               else:
                   dG_dt   = (GRID[g,t+1,s]-GRID[g,t-1,s]) / (2 * dthe)
                   d2G_dt2 = (-GRID[g,t+2,s]+16*GRID[g,t+1,s]-30*GRID[g,t,s]+16*GRID[g,t-1,s]-GRID[g,t-2,s]) / (12 * dthe**2)
                   d2G_dgt_numerator_t = -GRID[g,t+1,s] -GRID[g,t-1,s]


               # cross terms: they're 2?
               if g == 0 or t == 0:
                   d2G_dgt_numerator_cross_1 = 0
               else:
                   d2G_dgt_numerator_cross_1 = +GRID[g-1,t-1,s]

               if g == gamL-1 or t == theL-1:
                   d2G_dgt_numerator_cross_2 = 0
               else:
                   d2G_dgt_numerator_cross_2 = +GRID[g+1,t+1,s]


               d2G_dgt_numerator = d2G_dgt_numerator_g + d2G_dgt_numerator_t + d2G_dgt_numerator_cross_1 + d2G_dgt_numerator_cross_2 + 2*G
               d2G_dgt = d2G_dgt_numerator/(2*dgam*dthe)
               d2G_dtg = d2G_dgt

               # T elements
               Tgg = Km[g,t,4,0] * G + Km[g,t,4,1] * dG_dg + Km[g,t,4,2] * d2G_dg2
               Tgt = Km[g,t,5,0] * G + Km[g,t,5,1] * dG_dg + Km[g,t,5,2] * d2G_dgt
               Ttg = Km[g,t,7,0] * G + Km[g,t,7,1] * dG_dt + Km[g,t,7,2] * d2G_dtg
               Ttt = Km[g,t,8,0] * G + Km[g,t,8,1] * dG_dt + Km[g,t,8,2] * d2G_dt2

               Ttot = (Tgg + Tgt + Ttg + Ttt)
               Vtot = V * G

               # loop and sum on other states.
               Mtot = 0

               for d in range(nstates): # state s outer loop, state d inner loop
                   for carte in range(3): # carte is 'cartesian', meaning 0,1,2 -> x,y,z
                       Mtot = Mtot - ((pulseV[carte] * Dm[g,t,carte,s,d] ) * GRID[g,t,d])

               if selector == 1:
                   new[g,t,s] = I * (Ttot+Vtot+Mtot)
               else:
                   kinS[g,t,s] = Ttot
                   potS[g,t,s] = Vtot

    if selector == 1:
        return(new)
    else:
        return(kinS,potS)

#############
# 1D in phi #
#############

def Cenergy_1D_Phi(t,GRID,inp):
    '''wrapper for 1d in phi'''
    #print('ENERGY -> t: {} , wf inside: {}'.format(t,GRID.shape))
    return np.asarray(Cderivative1D_Phi_Mu(t,GRID,inp,0))

def Cderivative_1D_Phi(t,GRID,inp):
    '''wrapper for 1d in phi'''
    #print('PROPAG -> t: {} , wf inside: {}'.format(t,GRID.shape))
    return np.asarray(Cderivative1D_Phi_Mu(t,GRID,inp,1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Cderivative1D_Phi_Mu(double time, double complex [:,:] GRID,dict inp, int selector):
    '''
    derivative done for a 1D grid in Phi
    t :: Double -> time
    GRID :: np.array[:,:] <- wavefunction[phis,states]
    inp :: dictionary of various inputs
    '''
    cdef:
        int s,p,phiL=inp['phiL'],nstates=inp['nstates']
        int d,carte
        double dphi=inp['dphi'],V
        double [:,:] Vm = inp['potCube']
        double [:,:,:] Km = inp['kinCube']
        double [:,:,:,:] Dm = inp['dipCube']
        double [:] pulseV
        double complex [:,:] new, kinS, potS
        double complex I = -1j
        double complex dG_dp, d2G_dp2, G
        double complex Tpp
        double complex Ttot,Vtot,Mtot

    new = np.empty_like(GRID)
    kinS = np.empty_like(GRID)
    potS = np.empty_like(GRID)

    pulseV = np.empty((3))

    pulseV[0] = pulZe(time,inp['pulseX'])
    pulseV[1] = pulZe(time,inp['pulseY'])
    pulseV[2] = pulZe(time,inp['pulseZ'])

    #for s in range(nstates):
    for s in range(nstates):
        for p in range(phiL):
            G = GRID[p,s]
            V = Vm[p,s]

            # derivatives in phi
            if p == 0:
                dG_dp   = (GRID[p+1,s]) / (2 * dphi)
                d2G_dp2 = (-GRID[p+2,s]+16*GRID[p+1,s]-30*GRID[p,s]) / (12 * dphi**2)

            elif p == 1:
                dG_dp   = (GRID[p+1,s]-GRID[p-1,s]) / (2 * dphi)
                d2G_dp2 = (-GRID[p+2,s]+16*GRID[p+1,s]-30*GRID[p,s]+16*GRID[p-1,s]) / (12 * dphi**2)

            elif p == phiL-2:
                dG_dp   = (GRID[p+1,s]-GRID[p-1,s]) / (2 * dphi)
                d2G_dp2 = (+16*GRID[p+1,s]-30*GRID[p,s]+16*GRID[p-1,s]-GRID[p-2,s]) / (12 * dphi**2)

            elif p == phiL-1:
                dG_dp   = (-GRID[p-1,s]) / (2 * dphi)
                d2G_dp2 = (-30*GRID[p,s]+16*GRID[p-1,s]-GRID[p-2,s]) / (12 * dphi**2)

            else:
                dG_dp   = (GRID[p+1,s]-GRID[p-1,s]) / (2 * dphi)
                d2G_dp2 = (-GRID[p+2,s]+16*GRID[p+1,s]-30*GRID[p,s]+16*GRID[p-1,s]-GRID[p-2,s]) / (12 * dphi**2)


            # T elements (9)
            Tpp = Km[p,0,0] * G + Km[p,0,1] * dG_dp + Km[p,0,2] * d2G_dp2

            Ttot = Tpp
            Vtot = V * G

            # loop and sum on other states.
            Mtot = 0

            for d in range(nstates): # state s is where the outer loop is, d is where the inner loop is.
                for carte in range(3): # carte is 'cartesian', meaning 0,1,2 -> x,y,z
                    Mtot = Mtot - ((pulseV[carte] * Dm[p,carte,s,d] ) * GRID[p,d])

            if selector == 1:
                new[p,s] = I * (Ttot+Vtot+Mtot)
            else:
                kinS[p,s] = Ttot
                potS[p,s] = Vtot
    if selector == 1:
        return(new)
    else:
        return(kinS,potS)


#############
# 1D in gam #
#############

def Cenergy_1D_Gam(t,GRID,inp):
    '''wrapper for 1d in gamma'''
    return np.asarray(Cderivative1D_Gam_Mu(t,GRID,inp,0))

def Cderivative_1D_Gam(t,GRID,inp):
    '''wrapper for 1d in gamma'''
    return np.asarray(Cderivative1D_Gam_Mu(t,GRID,inp,1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Cderivative1D_Gam_Mu(double time, double complex [:,:] GRID,dict inp, int selector):
    '''
    derivative done for a 1D grid in Gamma
    t :: Double -> time
    GRID :: np.array[:,:] <- wavefunction[gams,states]
    inp :: dictionary of various inputs
    '''
    cdef:
        int s,g,gamL=inp['gamL'],nstates=inp['nstates']
        int d,carte
        double dgam=inp['dgam'],V
        double [:,:] Vm = inp['potCube']
        double [:,:,:] Km = inp['kinCube']
        double [:,:,:,:] Dm = inp['dipCube']
        double [:] pulseV
        double complex [:,:] new, kinS, potS
        double complex I = -1j
        double complex dG_dg, d2G_dg2, G
        double complex Tgg
        double complex Ttot,Vtot,Mtot

    new = np.empty_like(GRID)
    kinS = np.empty_like(GRID)
    potS = np.empty_like(GRID)

    pulseV = np.empty((3))

    pulseV[0] = pulZe(time,inp['pulseX'])
    pulseV[1] = pulZe(time,inp['pulseY'])
    pulseV[2] = pulZe(time,inp['pulseZ'])

    #for s in range(nstates):
    for s in range(nstates):
        for g in range(gamL):
            G = GRID[g,s]
            V = Vm[g,s]

            # derivatives in gam
            if g == 0:
                dG_dg   = (GRID[g+1,s]) / (2 * dgam)
                d2G_dg2 = (-GRID[g+2,s]+16*GRID[g+1,s]-30*GRID[g,s]) / (12 * dgam**2)

            elif g == 1:
                dG_dg   = (GRID[g+1,s]-GRID[g-1,s]) / (2 * dgam)
                d2G_dg2 = (-GRID[g+2,s]+16*GRID[g+1,s]-30*GRID[g,s]+16*GRID[g-1,s]) / (12 * dgam**2)

            elif g == gamL-2:
                dG_dg   = (GRID[g+1,s]-GRID[g-1,s]) / (2 * dgam)
                d2G_dg2 = (+16*GRID[g+1,s]-30*GRID[g,s]+16*GRID[g-1,s]-GRID[g-2,s]) / (12 * dgam**2)

            elif g == gamL-1:
                dG_dg   = (-GRID[g-1,s]) / (2 * dgam)
                d2G_dg2 = (-30*GRID[g,s]+16*GRID[g-1,s]-GRID[g-2,s]) / (12 * dgam**2)

            else:
                dG_dg   = (GRID[g+1,s]-GRID[g-1,s]) / (2 * dgam)
                d2G_dg2 = (-GRID[g+2,s]+16*GRID[g+1,s]-30*GRID[g,s]+16*GRID[g-1,s]-GRID[g-2,s]) / (12 * dgam**2)


            Tgg = Km[g,4,0] * G + Km[g,4,1] * dG_dg + Km[g,4,2] * d2G_dg2

            Ttot = Tgg
            Vtot = V * G

            # loop and sum on other states.
            Mtot = 0

            for d in range(nstates): # state s is where the outer loop is, d is where the inner loop is.
                for carte in range(3): # carte is 'cartesian', meaning 0,1,2 -> x,y,z
                    Mtot = Mtot - ((pulseV[carte] * Dm[g,carte,s,d] ) * GRID[g,d])

            if selector == 1:
                new[g,s] = I * (Ttot+Vtot+Mtot)
            else:
                kinS[g,s] = Ttot
                potS[g,s] = Vtot
    if selector == 1:
        return(new)
    else:
        return(kinS,potS)


#############
# 1D in the #
#############

def Cenergy_1D_The(t,GRID,inp):
    '''wrapper for 1d in theta'''
    return np.asarray(Cderivative1D_The_Mu(t,GRID,inp,0))

def Cderivative_1D_The(t,GRID,inp):
    '''wrapper for 1d in theta'''
    return np.asarray(Cderivative1D_The_Mu(t,GRID,inp,1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Cderivative1D_The_Mu(double time, double complex [:,:] GRID,dict inp, int selector):
    '''
    derivative done for a 1D grid in Theta
    t :: Double -> time
    GRID :: np.array[:,:] <- wavefunction[thes,states]
    inp :: dictionary of various inputs
    '''
    cdef:
        int s,t,theL=inp['theL'],nstates=inp['nstates']
        int d,carte
        double dthe=inp['dthe'],V
        double [:,:] Vm = inp['potCube']
        double [:,:,:] Km = inp['kinCube']
        double [:,:,:,:] Dm = inp['dipCube']
        double [:] pulseV
        double complex [:,:] new, kinS, potS
        double complex I = -1j
        double complex dG_dt, d2G_dt2, G
        double complex Ttt
        double complex Ttot,Vtot,Mtot

    new = np.empty_like(GRID)
    kinS = np.empty_like(GRID)
    potS = np.empty_like(GRID)

    pulseV = np.empty((3))

    pulseV[0] = pulZe(time,inp['pulseX'])
    pulseV[1] = pulZe(time,inp['pulseY'])
    pulseV[2] = pulZe(time,inp['pulseZ'])

    #for s in range(nstates):
    for s in range(nstates):
        for t in range(theL):
            G = GRID[t,s]
            V = Vm[t,s]

            # derivatives in theta
            if t == 0:
                dG_dt   = (GRID[t+1,s]) / (2 * dthe)
                d2G_dt2 = (-GRID[t+2,s]+16*GRID[t+1,s]-30*GRID[t,s]) / (12 * dthe**2)

            elif t == 1:
                dG_dt   = (GRID[t+1,s]-GRID[t-1,s]) / (2 * dthe)
                d2G_dt2 = (-GRID[t+2,s]+16*GRID[t+1,s]-30*GRID[t,s]+16*GRID[t-1,s]) / (12 * dthe**2)

            elif t == theL-2:
                dG_dt   = (GRID[t+1,s]-GRID[t-1,s]) / (2 * dthe)
                d2G_dt2 = (+16*GRID[t+1,s]-30*GRID[t,s]+16*GRID[t-1,s]-GRID[t-2,s]) / (12 * dthe**2)

            elif t == theL-1:
                dG_dt   = (-GRID[t-1,s]) / (2 * dthe)
                d2G_dt2 = (-30*GRID[t,s]+16*GRID[t-1,s]-GRID[t-2,s]) / (12 * dthe**2)

            else:
                dG_dt   = (GRID[t+1,s]-GRID[t-1,s]) / (2 * dthe)
                d2G_dt2 = (-GRID[t+2,s]+16*GRID[t+1,s]-30*GRID[t,s]+16*GRID[t-1,s]-GRID[t-2,s]) / (12 * dthe**2)


            # T elements (9)
            Ttt = Km[t,8,0] * G + Km[t,8,1] * dG_dt + Km[t,8,2] * d2G_dt2

            Ttot = Ttt
            Vtot = V * G

            # loop and sum on other states.
            Mtot = 0

            for d in range(nstates): # state s is where the outer loop is, d is where the inner loop is.
                for carte in range(3): # carte is 'cartesian', meaning 0,1,2 -> x,y,z
                    Mtot = Mtot - ((pulseV[carte] * Dm[t,carte,s,d] ) * GRID[t,d])

            if selector == 1:
                new[t,s] = I * (Ttot+Vtot+Mtot)
            else:
                kinS[t,s] = Ttot
                potS[t,s] = Vtot
    if selector == 1:
        return(new)
    else:
        return(kinS,potS)

