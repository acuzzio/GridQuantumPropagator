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

def version_Cpropagator():
    return('0.0.0026 the one with FT')


def fft_artisanal(time,signal):
    '''
    This performs fourier transform of the signal at time.
    time :: np.array(double) <- AU
    signal :: np.array(double) <- AU

    it returns:
    fft_array :: np.array(complex) <- AU
    np.array(freq) :: np.array(double) <- AU
    '''
    dt = time[1] - time[0]
    nstep = time.size
    all_time = nstep * dt
    sx = -np.pi/dt
    dx = np.pi/dt
    dw = (2 * np.pi)/all_time
    freq = np.arange(sx,dx,dw)
    fft_array, freq = fft_c(time, signal, freq, dt, nstep)
    return (fft_array, np.array(freq))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef fft_c(double [:] time, double [:] signal, double [:] freq, int dt, int nstep):
    # this is a simple FT using complex exp function cexp
    cdef:
        int k,j
        double complex I = -1j

    fft_array = np.zeros(nstep, dtype=complex)

    for k in range(nstep):
        for j in range(nstep):

            fft_array[k] = fft_array[k] + cexp(I * freq[k] * time[j]) * signal[j]

    return(fft_array, freq)

def Crk4Ene3d(f, t, y, inp):
    '''
    Runge Kutta with 3d grid as y
    f -> function that returns the derivative value
    t -> time at which the derivative is calculated
    y -> the numpy array
    inp -> dictionaries with immutable objects
    '''
    h = inp['dt']
    k1 = h * f(t, y, inp)
    k2 = h * f(t + 0.5 * h, y + 0.5 * k1, inp)
    k3 = h * f(t + 0.5 * h, y + 0.5 * k2, inp)
    k4 = h * f(t + h, y + k3, inp)
    return y + (k1 + k2 + k2 + k3 + k3 + k4) / 6

def pulZe(t, param_Pulse):
    '''
    Pulse function
    it works both with an array of time and a single time value
    '''
    Ed,omega,sigma,phase,t0 = param_Pulse
    num = (t-t0)**2
    den = 2.0*(sigma**2)
    if (den == 0):
        if type(t) == float:
            result = 0.0
        else:
            result = np.zeros_like(t)
    else:
        result = Ed * (np.cos(omega*(t-t0)+phase)) * np.exp(-num/den)
    return result

def pulZe2(t, param_Pulse):
    '''
    Pulse function
    it works both with an array of time and a single time value
    This one is the new kind of pulse, which is consistent in energy space
    '''
    Ed,omega,sigma,phi,t0 = param_Pulse
    num = (t-t0)**2
    den = 2*(sigma**2)

    if (den == 0):
        result = 0.0
    else:
        num2 = np.sin(omega*(t-t0) + phi) * (t-t0)
        den2 = omega * sigma**2
        result = Ed *  np.exp(-num/den)* (np.cos(omega*(t-t0) + phi) - num2/den2 )
    return result

def calculate_dipole_fast_wrapper(wf, all_h5_dict, default_tuple_for_cube=None):
    '''
    quick dipole calculation for WF
    '''
    pL, gL, tL, nstates = wf.shape

    default_tuple_for_cube = default_tuple_for_cube or None

    if default_tuple_for_cube == None:
        pmin = 15
        pmax = pL-15
        gmin = 15
        gmax = gL-15
        tmin = 30
        tmax = tL-30
    else:
        pmin,pmax,gmin,gmax,tmin,tmax = default_tuple_for_cube

    dipoles = all_h5_dict['dipCube']
    return calculate_dipole_fast(wf, dipoles, nstates, pmin, pmax, gmin, gmax, tmin, tmax)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef calculate_dipole_fast(double complex [:,:,:,:] wf, double [:,:,:,:,:,:] dipoles, int nstates, int pmin, int pmax, int gmin, int gmax, int tmin, int tmax):
    '''
    dipole fast calculation
    '''
    cdef:
        int p,g,t,i,j,index
        double xd,yd,zd,abs2_grid
        double [:] out_of_diagonal_x, out_of_diagonal_y, out_of_diagonal_z
        double [:] diagonal_x, diagonal_y, diagonal_z

    diagonal_x = np.zeros(nstates)
    diagonal_y = np.zeros(nstates)
    diagonal_z = np.zeros(nstates)
    out_of_diagonal_x = np.zeros(((nstates*nstates-nstates)/2))
    out_of_diagonal_y = np.zeros(((nstates*nstates-nstates)/2))
    out_of_diagonal_z = np.zeros(((nstates*nstates-nstates)/2))

    xd = yd = zd = 0.0
    for p in range(pmin, pmax):
        for g in range(gmin, gmax):
            for t in range(tmin, tmax):
                for i in range(nstates):
                    for j in range(i,nstates):
                        index = (nstates*(nstates-1)/2) - (nstates-i)*((nstates-i)-1)/2 + j - i - 1
                        if j != i:
                            xd = xd  + 2 * (wf[p,g,t,i].conjugate() * wf[p,g,t,j] * dipoles[p,g,t,0,i,j]).real
                            yd = yd  + 2 * (wf[p,g,t,i].conjugate() * wf[p,g,t,j] * dipoles[p,g,t,1,i,j]).real
                            zd = zd  + 2 * (wf[p,g,t,i].conjugate() * wf[p,g,t,j] * dipoles[p,g,t,2,i,j]).real
                            out_of_diagonal_x[index] = out_of_diagonal_x[index] + 2 * (wf[p,g,t,i].conjugate() * wf[p,g,t,j] * dipoles[p,g,t,0,i,j]).real
                            out_of_diagonal_y[index] = out_of_diagonal_y[index] + 2 * (wf[p,g,t,i].conjugate() * wf[p,g,t,j] * dipoles[p,g,t,1,i,j]).real
                            out_of_diagonal_z[index] = out_of_diagonal_z[index] + 2 * (wf[p,g,t,i].conjugate() * wf[p,g,t,j] * dipoles[p,g,t,2,i,j]).real
                        else:
                            abs2_grid = wf[p,g,t,i].real**2 + wf[p,g,t,i].imag**2
                            xd = xd  + (abs2_grid * dipoles[p,g,t,0,i,i])
                            yd = yd  + (abs2_grid * dipoles[p,g,t,1,i,i])
                            zd = zd  + (abs2_grid * dipoles[p,g,t,2,i,i])
                            diagonal_x[i] = diagonal_x[i] + (abs2_grid * dipoles[p,g,t,0,i,i])
                            diagonal_y[i] = diagonal_y[i] + (abs2_grid * dipoles[p,g,t,1,i,i])
                            diagonal_z[i] = diagonal_z[i] + (abs2_grid * dipoles[p,g,t,2,i,i])
    return(xd,yd,zd,diagonal_x,diagonal_y,diagonal_z,out_of_diagonal_x,out_of_diagonal_y,out_of_diagonal_z)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
def CextractMomentum3d(double complex [:,:,:,:] GRID, dict inp):
    '''
    Function that extract momentum from a wavefunction
    GRID :: np.array[:,:,:,:] <- wavefunction[phis,gams,thes,states]
    inp :: dictionary of various inputs
    '''
    cdef:
        int s,p,g,t,phiL=inp['phiL'],gamL=inp['gamL'],theL=inp['theL'],nstates=inp['nstates']
        int tuPlL,s_p
        double dphi=inp['dphi'],dgam=inp['dgam'],dthe=inp['dthe']
        double complex [:,:,:,:] momSp,momSg,momSt
        double complex I = -1j
        double complex dG_dp_oth, dG_dg_oth, dG_dt_oth

    momSp = np.empty_like(GRID)
    momSg = np.empty_like(GRID)
    momSt = np.empty_like(GRID)

    tuPlL = nstates*phiL

    #for s_p in prange(tuPlL, nogil=True, schedule='dynamic'):
    for s_p in range(tuPlL):
       s = s_p // phiL
       p = s_p % phiL
       for g in range(gamL):
          for t in range(theL):
              # phi
              if p == 0:
                  dG_dp_oth = (                                                      (2.0/3)*GRID[p+1,g,t,s] + (-1.0/12)*GRID[p+2,g,t,s]) / dphi
              elif p == 1:
                  dG_dp_oth = (                           (-2.0/3)*GRID[p-1,g,t,s] + (2.0/3)*GRID[p+1,g,t,s] + (-1.0/12)*GRID[p+2,g,t,s]) / dphi
              elif p == phiL-2:
                  dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,s] + (-2.0/3)*GRID[p-1,g,t,s] + (2.0/3)*GRID[p+1,g,t,s]                            ) / dphi
              elif p == phiL-1:
                  dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,s] + (-2.0/3)*GRID[p-1,g,t,s]                                                      ) / dphi
              else:
                  dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,s] + (-2.0/3)*GRID[p-1,g,t,s] + (2.0/3)*GRID[p+1,g,t,s] + (-1.0/12)*GRID[p+2,g,t,s]) / dphi

              # gamma
              if g == 0:
                  dG_dg_oth = (                                                      (2.0/3)*GRID[p,g+1,t,s] + (-1.0/12)*GRID[p,g+2,t,s]) / dgam
              elif g == 1:
                  dG_dg_oth = (                           (-2.0/3)*GRID[p,g-1,t,s] + (2.0/3)*GRID[p,g+1,t,s] + (-1.0/12)*GRID[p,g+2,t,s]) / dgam
              elif g == gamL-2:
                  dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,s] + (-2.0/3)*GRID[p,g-1,t,s] + (2.0/3)*GRID[p,g+1,t,s]                            ) / dgam
              elif g == gamL-1:
                  dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,s] + (-2.0/3)*GRID[p,g-1,t,s]                                                      ) / dgam
              else:
                  dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,s] + (-2.0/3)*GRID[p,g-1,t,s] + (2.0/3)*GRID[p,g+1,t,s] + (-1.0/12)*GRID[p,g+2,t,s]) / dgam

              # theta
              if t == 0:
                  dG_dt_oth = (                                                      (2.0/3)*GRID[p,g,t+1,s] + (-1.0/12)*GRID[p,g,t+2,s]) / dthe
              elif t == 1:
                  dG_dt_oth = (                           (-2.0/3)*GRID[p,g,t-1,s] + (2.0/3)*GRID[p,g,t+1,s] + (-1.0/12)*GRID[p,g,t+2,s]) / dthe
              elif t == theL-2:
                  dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,s] + (-2.0/3)*GRID[p,g,t-1,s] + (2.0/3)*GRID[p,g,t+1,s]                            ) / dthe
              elif t == theL-1:
                  dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,s] + (-2.0/3)*GRID[p,g,t-1,s]                                                      ) / dthe
              else:
                  dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,s] + (-2.0/3)*GRID[p,g,t-1,s] + (2.0/3)*GRID[p,g,t+1,s] + (-1.0/12)*GRID[p,g,t+2,s]) / dthe

              momSp[p,g,t,s] = I * dG_dp_oth
              momSg[p,g,t,s] = I * dG_dg_oth
              momSt[p,g,t,s] = I * dG_dt_oth

    return(np.asarray(momSp),np.asarray(momSg),np.asarray(momSt))

def CextractEnergy3dMu(t,GRID,inp):
    '''wrapper for 3d integrator in Kinetic-Potential mode'''
    #print('ENERGY -> t: {} , wf inside: {}'.format(t,GRID.shape))
    return np.asarray(Cderivative3dMu_cyt(t,GRID,inp,0,1))

def Cderivative3dMu(t,GRID,inp):
    '''wrapper for 3d integrator'''
    #print('PROPAG -> t: {} , wf inside: {}'.format(t,GRID.shape))
    return np.asarray(Cderivative3dMu_cyt(t,GRID,inp,1,1))

def Cderivative3dMu_reverse_time(t,GRID,inp):
    '''wrapper for 3d integrator time reversed'''
    #print('PROPAG -> t: {} , wf inside: {}'.format(t,GRID.shape))
    return np.asarray(Cderivative3dMu_cyt(t,GRID,inp,1,-1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef Cderivative3dMu_cyt(double time, double complex [:,:,:,:] GRID, dict inp, int selector, int reverse_or_not):
    '''
    derivative done for a 3d Grid on all the coordinates
    t :: Double -> time
    GRID :: np.array[:,:,:,:] <- wavefunction[phis,gams,thes,states]
    inp :: dictionary of various inputs
    export OMP_NUM_THREADS=10
    '''
    cdef:
        int s,p,g,t,phiL=inp['phiL'],gamL=inp['gamL'],theL=inp['theL'],nstates=inp['nstates']
        int d,carte,tuPlL,s_p
        int [:,:] tuPl
        double dphi=inp['dphi'],dgam=inp['dgam'],dthe=inp['dthe'],V,Ab
        double [:,:,:,:] Vm = inp['potCube']
        double [:,:,:,:] absorb = inp['absorb']
        double [:,:,:,:,:] Km = inp['kinCube']
        double [:,:,:,:,:,:] Dm = inp['dipCube']
        double [:,:,:,:,:,:] Nm = inp['nacCube']
        double [:] pulseV
        double complex [:,:,:,:] new, kinS, potS, pulS, absS
        double complex I = -1j * reverse_or_not
        double complex dG_dp, d2G_dp2, dG_dg, d2G_dg2, dG_dt, d2G_dt2, G
        double complex dG_dp_oth, dG_dg_oth, dG_dt_oth
        double complex d2G_dcross_numerator_p,d2G_dcross_numerator_g,d2G_dcross_numerator_t
        double complex d2G_dpg_numerator_cross_1,d2G_dpg_numerator_cross_2,d2G_dpg_numerator
        double complex d2G_dpt_numerator_cross_1,d2G_dpt_numerator_cross_2,d2G_dpt_numerator
        double complex d2G_dgt_numerator_cross_1,d2G_dgt_numerator_cross_2,d2G_dgt_numerator
        double complex d2G_dpg,d2G_dpt,d2G_dgt,d2G_dgp,d2G_dtp,d2G_dtg
        double complex Tpp,Tpg,Tpt,Tgp,Tgg,Tgt,Ttp,Ttg,Ttt
        double complex Ttot,Vtot,Mtot,Ntot,Atot

    new = np.empty_like(GRID)
    kinS = np.empty_like(GRID)
    potS = np.empty_like(GRID)
    pulS = np.empty_like(GRID)
    absS = np.empty_like(GRID)

    pulseV = np.empty((3))

    pulseV[0] = pulZe2(time,inp['pulseX'])
    pulseV[1] = pulZe2(time,inp['pulseY'])
    pulseV[2] = pulZe2(time,inp['pulseZ'])

    #for s in range(nstates):

    tuPlL = nstates*phiL

    # parallel 8 version
    #for s in prange(nstates, nogil=True):   # first state loop.
    #    for p in range(phiL):

    # trying to get 16
    for s_p in prange(tuPlL, nogil=True):
    #for s_p in prange(tuPlL, nogil=True, schedule='dynamic'):
    #for s_p in range(tuPlL):
            s = s_p // phiL
            p = s_p % phiL
            for g in range(gamL):
               for t in range(theL):
                   G = GRID[p,g,t,s]
                   V = Vm[p,g,t,s]
                   Ab = absorb[p,g,t,s]

                   # derivatives in phi
                   if p == 0:
                       dG_dp   = ((2.0/3)*GRID[p+1,g,t,s]+(-1.0/12)*GRID[p+2,g,t,s]) / dphi
                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s]

                   elif p == 1:
                       dG_dp   = ((-2.0/3)*GRID[p-1,g,t,s]+(2.0/3)*GRID[p+1,g,t,s]+(-1.0/12)*GRID[p+2,g,t,s]) / dphi
                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]

                   elif p == phiL-2:
                       dG_dp   = ((1.0/12)*GRID[p-2,g,t,s]+(-2.0/3)*GRID[p-1,g,t,s]+(2.0/3)*GRID[p+1,g,t,s]) / dphi
                       d2G_dp2 = (+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]

                   elif p == phiL-1:
                       dG_dp   = ((1.0/12)*GRID[p-2,g,t,s]+(-2.0/3)*GRID[p-1,g,t,s]) / dphi
                       d2G_dp2 = (-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p-1,g,t,s]

                   else:
                       dG_dp   = ((1.0/12)*GRID[p-2,g,t,s]+(-2.0/3)*GRID[p-1,g,t,s]+(2.0/3)*GRID[p+1,g,t,s]+(-1.0/12)*GRID[p+2,g,t,s]) / dphi
                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]

                   # derivatives in gam
                   if g == 0:
                       dG_dg   = ((2.0/3)*GRID[p,g+1,t,s]+(-1.0/12)*GRID[p,g+2,t,s]) / dgam
                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s]

                   elif g == 1:
                       dG_dg   = ((-2.0/3)*GRID[p,g-1,t,s]+(2.0/3)*GRID[p,g+1,t,s]+(-1.0/12)*GRID[p,g+2,t,s]) / dgam
                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]

                   elif g == gamL-2:
                       dG_dg   = ((1.0/12)*GRID[p,g-2,t,s]+(-2.0/3)*GRID[p,g-1,t,s]+(2.0/3)*GRID[p,g+1,t,s]) / dgam
                       d2G_dg2 = (+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]

                   elif g == gamL-1:
                       dG_dg   = ((1.0/12)*GRID[p,g-2,t,s]+(-2.0/3)*GRID[p,g-1,t,s]) / dgam
                       d2G_dg2 = (-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g-1,t,s]

                   else:
                       dG_dg   = ((1.0/12)*GRID[p,g-2,t,s]+(-2.0/3)*GRID[p,g-1,t,s]+(2.0/3)*GRID[p,g+1,t,s]+(-1.0/12)*GRID[p,g+2,t,s]) / dgam
                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]

                   # derivatives in the
                   if t == 0:
                       dG_dt   = ((2.0/3)*GRID[p,g,t+1,s]+(-1.0/12)*GRID[p,g,t+2,s]) / dthe
                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s]

                   elif t == 1:
                       dG_dt   = ((-2.0/3)*GRID[p,g,t-1,s]+(2.0/3)*GRID[p,g,t+1,s]+(-1.0/12)*GRID[p,g,t+2,s]) / dthe
                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]

                   elif t == theL-2:
                       dG_dt   = ((1.0/12)*GRID[p,g,t-2,s]+(-2.0/3)*GRID[p,g,t-1,s]+(2.0/3)*GRID[p,g,t+1,s]) / dthe
                       d2G_dt2 = (+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]

                   elif t == theL-1:
                       dG_dt   = ((1.0/12)*GRID[p,g,t-2,s]+(-2.0/3)*GRID[p,g,t-1,s]) / dthe
                       d2G_dt2 = (-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
                       d2G_dcross_numerator_t = -GRID[p,g,t-1,s]

                   else:
                       dG_dt   = ((1.0/12)*GRID[p,g,t-2,s]+(-2.0/3)*GRID[p,g,t-1,s]+(2.0/3)*GRID[p,g,t+1,s]+(-1.0/12)*GRID[p,g,t+2,s]) / dthe
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
                   Tpt = Km[p,g,t,2,0] * G + Km[p,g,t,2,1] * dG_dp + Km[p,g,t,2,2] * d2G_dpt

                   Tgp = Km[p,g,t,3,0] * G + Km[p,g,t,3,1] * dG_dg + Km[p,g,t,3,2] * d2G_dgp
                   Tgg = Km[p,g,t,4,0] * G + Km[p,g,t,4,1] * dG_dg + Km[p,g,t,4,2] * d2G_dg2
                   Tgt = Km[p,g,t,5,0] * G + Km[p,g,t,5,1] * dG_dg + Km[p,g,t,5,2] * d2G_dgt

                   Ttp = Km[p,g,t,6,0] * G + Km[p,g,t,6,1] * dG_dt + Km[p,g,t,6,2] * d2G_dtp
                   Ttg = Km[p,g,t,7,0] * G + Km[p,g,t,7,1] * dG_dt + Km[p,g,t,7,2] * d2G_dtg
                   Ttt = Km[p,g,t,8,0] * G + Km[p,g,t,8,1] * dG_dt + Km[p,g,t,8,2] * d2G_dt2

                   Ttot = (Tpp + Tpg + Tpt + Tgp + Tgg + Tgt + Ttp + Ttg + Ttt)
                   Vtot = V * G
                   Atot = - Ab * G * 1j

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

                       if s < d:
                          # phi
                          if p == 0:
                              dG_dp_oth = (                                                      (2.0/3)*GRID[p+1,g,t,d] + (-1.0/12)*GRID[p+2,g,t,d])*Nm[p,g,t,s,d,0] / dphi
                          elif p == 1:
                              dG_dp_oth = (                           (-2.0/3)*GRID[p-1,g,t,d] + (2.0/3)*GRID[p+1,g,t,d] + (-1.0/12)*GRID[p+2,g,t,d])*Nm[p,g,t,s,d,0] / dphi
                          elif p == phiL-2:
                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d] + (-2.0/3)*GRID[p-1,g,t,d] + (2.0/3)*GRID[p+1,g,t,d]                            )*Nm[p,g,t,s,d,0] / dphi
                          elif p == phiL-1:
                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d] + (-2.0/3)*GRID[p-1,g,t,d]                                                      )*Nm[p,g,t,s,d,0] / dphi
                          else:
                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d] + (-2.0/3)*GRID[p-1,g,t,d] + (2.0/3)*GRID[p+1,g,t,d] + (-1.0/12)*GRID[p+2,g,t,d])*Nm[p,g,t,s,d,0] / dphi
                          # gamma
                          if g == 0:
                              dG_dg_oth = (                                                      (2.0/3)*GRID[p,g+1,t,d] + (-1.0/12)*GRID[p,g+2,t,d])*Nm[p,g,t,s,d,1] / dgam
                          elif g == 1:
                              dG_dg_oth = (                           (-2.0/3)*GRID[p,g-1,t,d] + (2.0/3)*GRID[p,g+1,t,d] + (-1.0/12)*GRID[p,g+2,t,d])*Nm[p,g,t,s,d,1] / dgam
                          elif g == gamL-2:
                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d] + (-2.0/3)*GRID[p,g-1,t,d] + (2.0/3)*GRID[p,g+1,t,d]                            )*Nm[p,g,t,s,d,1] / dgam
                          elif g == gamL-1:
                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d] + (-2.0/3)*GRID[p,g-1,t,d]                                                      )*Nm[p,g,t,s,d,1] / dgam
                          else:
                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d] + (-2.0/3)*GRID[p,g-1,t,d] + (2.0/3)*GRID[p,g+1,t,d] + (-1.0/12)*GRID[p,g+2,t,d])*Nm[p,g,t,s,d,1] / dgam
                          # theta
                          if t == 0:
                              dG_dt_oth = (                                                      (2.0/3)*GRID[p,g,t+1,d] + (-1.0/12)*GRID[p,g,t+2,d])*Nm[p,g,t,s,d,2] / dthe
                          elif t == 1:
                              dG_dt_oth = (                           (-2.0/3)*GRID[p,g,t-1,d] + (2.0/3)*GRID[p,g,t+1,d] + (-1.0/12)*GRID[p,g,t+2,d])*Nm[p,g,t,s,d,2] / dthe
                          elif t == theL-2:
                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d] + (-2.0/3)*GRID[p,g,t-1,d] + (2.0/3)*GRID[p,g,t+1,d]                            )*Nm[p,g,t,s,d,2] / dthe
                          elif t == theL-1:
                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d] + (-2.0/3)*GRID[p,g,t-1,d]                                                      )*Nm[p,g,t,s,d,2] / dthe
                          else:
                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d] + (-2.0/3)*GRID[p,g,t-1,d] + (2.0/3)*GRID[p,g,t+1,d] + (-1.0/12)*GRID[p,g,t+2,d])*Nm[p,g,t,s,d,2] / dthe
                       elif s > d:
                          # phi
                          if p == 0:
                              dG_dp_oth = (                                                                                          (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0] + (-1.0/12)*GRID[p+2,g,t,d]*Nm[p+2,g,t,s,d,0]) / dphi
                          elif p == 1:
                              dG_dp_oth = (                                             (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0] + (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0] + (-1.0/12)*GRID[p+2,g,t,d]*Nm[p+2,g,t,s,d,0]) / dphi
                          elif p == phiL-2:
                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d]*Nm[p-2,g,t,s,d,0] + (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0] + (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0]                                              ) / dphi
                          elif p == phiL-1:
                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d]*Nm[p-2,g,t,s,d,0] + (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0]                                                                                          ) / dphi
                          else:
                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d]*Nm[p-2,g,t,s,d,0] + (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0] + (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0] + (-1.0/12)*GRID[p+2,g,t,d]*Nm[p+2,g,t,s,d,0]) / dphi
                          # gam
                          if g == 0:
                              dG_dg_oth = (                                                                                          (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1] + (-1.0/12)*GRID[p,g+2,t,d]*Nm[p,g+2,t,s,d,1]) / dgam
                          elif g == 1:
                              dG_dg_oth = (                                             (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1] + (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1] + (-1.0/12)*GRID[p,g+2,t,d]*Nm[p,g+2,t,s,d,1]) / dgam
                          elif g == gamL-2:
                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d]*Nm[p,g-2,t,s,d,1] + (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1] + (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1]                                              ) / dgam
                          elif g == gamL-1:
                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d]*Nm[p,g-2,t,s,d,1] + (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1]                                                                                          ) / dgam
                          else:
                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d]*Nm[p,g-2,t,s,d,1] + (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1] + (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1] + (-1.0/12)*GRID[p,g+2,t,d]*Nm[p,g+2,t,s,d,1]) / dgam
                          # the
                          if t == 0:
                              dG_dt_oth = (                                                                                          (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2] + (-1.0/12)*GRID[p,g,t+2,d]*Nm[p,g,t+2,s,d,2]) / dthe
                          elif t == 1:
                              dG_dt_oth = (                                             (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2] + (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2] + (-1.0/12)*GRID[p,g,t+2,d]*Nm[p,g,t+2,s,d,2]) / dthe
                          elif t == theL-2:
                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d]*Nm[p,g,t-2,s,d,2] + (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2] + (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2]                                              ) / dthe
                          elif t == theL-1:
                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d]*Nm[p,g,t-2,s,d,2] + (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2]                                                                                          ) / dthe
                          else:
                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d]*Nm[p,g,t-2,s,d,2] + (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2] + (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2] + (-1.0/12)*GRID[p,g,t+2,d]*Nm[p,g,t+2,s,d,2]) / dthe
                       else:
                          dG_dp_oth = 0
                          dG_dg_oth = 0
                          dG_dt_oth = 0

                       Ntot = Ntot - (dG_dp_oth + dG_dg_oth + dG_dt_oth)

                   if selector == 1:
                       #new[p,g,t,s] = I * (Ttot + Vtot + Mtot + Ntot)
                       new[p,g,t,s] = I * (Ttot + Vtot + Atot + Mtot + Ntot)
                   else:
                       kinS[p,g,t,s] = Ttot + Ntot
                       potS[p,g,t,s] = Vtot
                       pulS[p,g,t,s] = Mtot
                       absS[p,g,t,s] = Atot

    if selector == 1:
        return(new)
    else:
        return(kinS,potS,pulS,absS)


# ATTEMPT TO VECTORIZE THE MAIN FUNCION

#def transform_index(a,b,c,d):
#    al = 7
#    bl = 7
#    cl = 7
#    dl = 2
#    return(a*(bl*cl*dl)+b*(cl*dl)+c*(dl)+d)
#
#
#cdef Checking_addresses(double time, double complex [:,:,:,:] GRID, dict inp, int selector):
#    '''
#    derivative done for a 3d Grid on all the coordinates
#    t :: Double -> time
#    GRID :: np.array[:,:,:,:] <- wavefunction[phis,gams,thes,states]
#    inp :: dictionary of various inputs
#    export OMP_NUM_THREADS=10
#    '''
#    cdef:
#        int s,p,g,t,phiL=inp['phiL'],gamL=inp['gamL'],theL=inp['theL'],nstates=inp['nstates']
#        int d,carte,tuPlL,s_p
#        int [:,:] tuPl
#        double dphi=inp['dphi'],dgam=inp['dgam'],dthe=inp['dthe'],V,Ab
#        double [:,:,:,:] Vm = inp['potCube']
#        double [:,:,:,:] absorb = inp['absorb']
#        double [:,:,:,:,:] Km = inp['kinCube']
#        double [:,:,:,:,:,:] Dm = inp['dipCube']
#        double [:,:,:,:,:,:] Nm = inp['nacCube']
#        double [:] pulseV
#        double complex [:,:,:,:] new, kinS, potS, pulS, absS
#        double complex I = -1j
#        double complex dG_dp, d2G_dp2, dG_dg, d2G_dg2, dG_dt, d2G_dt2, G
#        double complex dG_dp_oth, dG_dg_oth, dG_dt_oth
#        double complex d2G_dcross_numerator_p,d2G_dcross_numerator_g,d2G_dcross_numerator_t
#        double complex d2G_dpg_numerator_cross_1,d2G_dpg_numerator_cross_2,d2G_dpg_numerator
#        double complex d2G_dpt_numerator_cross_1,d2G_dpt_numerator_cross_2,d2G_dpt_numerator
#        double complex d2G_dgt_numerator_cross_1,d2G_dgt_numerator_cross_2,d2G_dgt_numerator
#        double complex d2G_dpg,d2G_dpt,d2G_dgt,d2G_dgp,d2G_dtp,d2G_dtg
#        double complex Tpp,Tpg,Tpt,Tgp,Tgg,Tgt,Ttp,Ttg,Ttt
#        double complex Ttot,Vtot,Mtot,Ntot,Atot
#
#    new = np.empty_like(GRID)
#    kinS = np.empty_like(GRID)
#    potS = np.empty_like(GRID)
#    pulS = np.empty_like(GRID)
#    absS = np.empty_like(GRID)
#
#    pulseV = np.empty((3))
#
#    pulseV[0] = pulZe2(time,inp['pulseX'])
#    pulseV[1] = pulZe2(time,inp['pulseY'])
#    pulseV[2] = pulZe2(time,inp['pulseZ'])
#
#    #for s in range(nstates):
#
#    tuPlL = nstates*phiL
#
#    big_values = np.zeros(phiL * gamL * theL * nstates)
#    big_i = np.zeros(phiL * gamL * theL * nstates)
#    big_j = np.zeros(phiL * gamL * theL * nstates)
#
#    #for p in range(phiL):
#    #    for g in range(gamL):
#    #       for t in range(theL):
#    #           for s in range(nstates):   # first state loop.
#    for p in 0:
#        for g in 3:
#           for t in 4:
#               for s in 1:   # first state loop.
#                   V = Vm[p,g,t,s]
#                   Ab = absorb[p,g,t,s]
#                   row_address = transform_index(p,g,t,s)
#
#                   # derivatives in phi
#                   if p == 0:
#                       dG_dp   = ((2.0/3)*GRID[p+1,g,t,s]+(-1.0/12)*GRID[p+2,g,t,s]) / dphi
#
#                       # HERE HERE the transform function
#                       value = (2.0/3) / dphi * (Km[p,g,t,0,1] + Km[p,g,t,1,1] + Km[p,g,t,2,1])
#                       address = transform_index(p+1,g,t,s)
#                       # the number value should be at indexes (row_address,address) in the huge i,j matrix
#                       value2 = (-1.0/12) * Km[p,g,t,0,1] / dphi
#                       address2 =  transform_index(p+2,g,t,s)
#
#                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]) / (12 * dphi**2)
#                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s]
#
#                   elif p == 1:
#                       dG_dp   = ((-2.0/3)*GRID[p-1,g,t,s]+(2.0/3)*GRID[p+1,g,t,s]+(-1.0/12)*GRID[p+2,g,t,s]) / dphi
#                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]) / (12 * dphi**2)
#                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]
#
#                   elif p == phiL-2:
#                       dG_dp   = ((1.0/12)*GRID[p-2,g,t,s]+(-2.0/3)*GRID[p-1,g,t,s]+(2.0/3)*GRID[p+1,g,t,s]) / dphi
#                       d2G_dp2 = (+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
#                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]
#
#                   elif p == phiL-1:
#                       dG_dp   = ((1.0/12)*GRID[p-2,g,t,s]+(-2.0/3)*GRID[p-1,g,t,s]) / dphi
#                       d2G_dp2 = (-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
#                       d2G_dcross_numerator_p = -GRID[p-1,g,t,s]
#
#                   else:
#                       dG_dp   = ((1.0/12)*GRID[p-2,g,t,s]+(-2.0/3)*GRID[p-1,g,t,s]+(2.0/3)*GRID[p+1,g,t,s]+(-1.0/12)*GRID[p+2,g,t,s]) / dphi
#                       d2G_dp2 = (-GRID[p+2,g,t,s]+16*GRID[p+1,g,t,s]-30*GRID[p,g,t,s]+16*GRID[p-1,g,t,s]-GRID[p-2,g,t,s]) / (12 * dphi**2)
#                       d2G_dcross_numerator_p = -GRID[p+1,g,t,s] -GRID[p-1,g,t,s]
#
#                   # derivatives in gam
#                   if g == 0:
#                       dG_dg   = ((2.0/3)*GRID[p,g+1,t,s]+(-1.0/12)*GRID[p,g+2,t,s]) / dgam
#                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]) / (12 * dgam**2)
#                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s]
#
#                   elif g == 1:
#                       dG_dg   = ((-2.0/3)*GRID[p,g-1,t,s]+(2.0/3)*GRID[p,g+1,t,s]+(-1.0/12)*GRID[p,g+2,t,s]) / dgam
#                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]) / (12 * dgam**2)
#                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]
#
#                   elif g == gamL-2:
#                       dG_dg   = ((1.0/12)*GRID[p,g-2,t,s]+(-2.0/3)*GRID[p,g-1,t,s]+(2.0/3)*GRID[p,g+1,t,s]) / dgam
#                       d2G_dg2 = (+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
#                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]
#
#                   elif g == gamL-1:
#                       dG_dg   = ((1.0/12)*GRID[p,g-2,t,s]+(-2.0/3)*GRID[p,g-1,t,s]) / dgam
#                       d2G_dg2 = (-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
#                       d2G_dcross_numerator_g = -GRID[p,g-1,t,s]
#
#                   else:
#                       dG_dg   = ((1.0/12)*GRID[p,g-2,t,s]+(-2.0/3)*GRID[p,g-1,t,s]+(2.0/3)*GRID[p,g+1,t,s]+(-1.0/12)*GRID[p,g+2,t,s]) / dgam
#                       d2G_dg2 = (-GRID[p,g+2,t,s]+16*GRID[p,g+1,t,s]-30*GRID[p,g,t,s]+16*GRID[p,g-1,t,s]-GRID[p,g-2,t,s]) / (12 * dgam**2)
#                       d2G_dcross_numerator_g = -GRID[p,g+1,t,s] -GRID[p,g-1,t,s]
#
#                   # derivatives in the
#                   if t == 0:
#                       dG_dt   = ((2.0/3)*GRID[p,g,t+1,s]+(-1.0/12)*GRID[p,g,t+2,s]) / dthe
#                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]) / (12 * dthe**2)
#                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s]
#
#                   elif t == 1:
#                       dG_dt   = ((-2.0/3)*GRID[p,g,t-1,s]+(2.0/3)*GRID[p,g,t+1,s]+(-1.0/12)*GRID[p,g,t+2,s]) / dthe
#                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]) / (12 * dthe**2)
#                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]
#
#                   elif t == theL-2:
#                       dG_dt   = ((1.0/12)*GRID[p,g,t-2,s]+(-2.0/3)*GRID[p,g,t-1,s]+(2.0/3)*GRID[p,g,t+1,s]) / dthe
#                       d2G_dt2 = (+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
#                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]
#
#                   elif t == theL-1:
#                       dG_dt   = ((1.0/12)*GRID[p,g,t-2,s]+(-2.0/3)*GRID[p,g,t-1,s]) / dthe
#                       d2G_dt2 = (-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
#                       d2G_dcross_numerator_t = -GRID[p,g,t-1,s]
#
#                   else:
#                       dG_dt   = ((1.0/12)*GRID[p,g,t-2,s]+(-2.0/3)*GRID[p,g,t-1,s]+(2.0/3)*GRID[p,g,t+1,s]+(-1.0/12)*GRID[p,g,t+2,s]) / dthe
#                       d2G_dt2 = (-GRID[p,g,t+2,s]+16*GRID[p,g,t+1,s]-30*GRID[p,g,t,s]+16*GRID[p,g,t-1,s]-GRID[p,g,t-2,s]) / (12 * dthe**2)
#                       d2G_dcross_numerator_t = -GRID[p,g,t+1,s] -GRID[p,g,t-1,s]
#
#
#                   # cross terms: they're thousands... 
#                   if p == 0 or g == 0:
#                       d2G_dpg_numerator_cross_1 = 0
#                   else:
#                       d2G_dpg_numerator_cross_1 = +GRID[p-1,g-1,t,s]
#                   if p == phiL-1 or g == gamL-1:
#                       d2G_dpg_numerator_cross_2 = 0
#                   else:
#                       d2G_dpg_numerator_cross_2 = +GRID[p+1,g+1,t,s]
#
#                   if p == 0 or t == 0:
#                       d2G_dpt_numerator_cross_1 = 0
#                   else:
#                       d2G_dpt_numerator_cross_1 = +GRID[p-1,g,t-1,s]
#                   if p == phiL-1 or t == theL-1:
#                       d2G_dpt_numerator_cross_2 = 0
#                   else:
#                       d2G_dpt_numerator_cross_2 = +GRID[p+1,g,t+1,s]
#
#                   if g == 0 or t == 0:
#                       d2G_dgt_numerator_cross_1 = 0
#                   else:
#                       d2G_dgt_numerator_cross_1 = +GRID[p,g-1,t-1,s]
#                   if g == gamL-1 or t == theL-1:
#                       d2G_dgt_numerator_cross_2 = 0
#                   else:
#                       d2G_dgt_numerator_cross_2 = +GRID[p,g+1,t+1,s]
#
#                   # triple 0 or triplelast, we DO NOT NEED these term, as ANY of my terms in the kinetic energy depends on displacements along all three 
#                   # coordinates... thus, no special cases where (p ==o or g == 0 or t == 0)
#
#                   d2G_dpg_numerator = d2G_dcross_numerator_p + d2G_dcross_numerator_g + d2G_dpg_numerator_cross_1 + d2G_dpg_numerator_cross_2 + 2*G
#                   d2G_dpt_numerator = d2G_dcross_numerator_p + d2G_dcross_numerator_t + d2G_dpt_numerator_cross_1 + d2G_dpt_numerator_cross_2 + 2*G
#                   d2G_dgt_numerator = d2G_dcross_numerator_g + d2G_dcross_numerator_t + d2G_dgt_numerator_cross_1 + d2G_dgt_numerator_cross_2 + 2*G
#
#                   d2G_dpg = d2G_dpg_numerator/(2*dphi*dgam)
#                   d2G_dpt = d2G_dpt_numerator/(2*dphi*dthe)
#                   d2G_dgt = d2G_dgt_numerator/(2*dgam*dthe)
#                   d2G_dgp = d2G_dpg
#                   d2G_dtp = d2G_dpt
#                   d2G_dtg = d2G_dgt
#
#                   # T elements (9)
#                   Tpp = Km[p,g,t,0,0] * G + Km[p,g,t,0,1] * dG_dp + Km[p,g,t,0,2] * d2G_dp2
#                   Tpg = Km[p,g,t,1,0] * G + Km[p,g,t,1,1] * dG_dp + Km[p,g,t,1,2] * d2G_dpg
#                   Tpt = Km[p,g,t,2,0] * G + Km[p,g,t,2,1] * dG_dp + Km[p,g,t,2,2] * d2G_dpt
#
#                   Tgp = Km[p,g,t,3,0] * G + Km[p,g,t,3,1] * dG_dg + Km[p,g,t,3,2] * d2G_dgp
#                   Tgg = Km[p,g,t,4,0] * G + Km[p,g,t,4,1] * dG_dg + Km[p,g,t,4,2] * d2G_dg2
#                   Tgt = Km[p,g,t,5,0] * G + Km[p,g,t,5,1] * dG_dg + Km[p,g,t,5,2] * d2G_dgt
#
#                   Ttp = Km[p,g,t,6,0] * G + Km[p,g,t,6,1] * dG_dt + Km[p,g,t,6,2] * d2G_dtp
#                   Ttg = Km[p,g,t,7,0] * G + Km[p,g,t,7,1] * dG_dt + Km[p,g,t,7,2] * d2G_dtg
#                   Ttt = Km[p,g,t,8,0] * G + Km[p,g,t,8,1] * dG_dt + Km[p,g,t,8,2] * d2G_dt2
#
#                   Ttot = (Tpp + Tpg + Tpt + Tgp + Tgg + Tgt + Ttp + Ttg + Ttt)
#                   Vtot = V * G
#                   Atot = - Ab * G * 1j
#
#                   # loop and sum on other states.
#                   Mtot = 0
#                   Ntot = 0
#
#                   # this is state S with all the others. Summing up.
#
#                   for d in range(nstates): # state s is where the outer loop is, d is where the inner loop is.
#                       for carte in range(3): # carte is 'cartesian', meaning 0,1,2 -> x,y,z
#                           # Mtot += -(pulseV[carte] * D[carte,s,d] ) * GRID[p,g,t,d]
#                           # parallel version DOES NOT want += (better, it does not consider += as
#                           # the serial version would. += works on shared reduction
#                           # variables inside of the prange() loop
#                           Mtot = Mtot - ((pulseV[carte] * Dm[p,g,t,carte,s,d] ) * GRID[p,g,t,d])
#
#                       # NAC calculation
#
#                       if s < d:
#                          # phi
#                          if p == 0:
#                              dG_dp_oth = (                                                      (2.0/3)*GRID[p+1,g,t,d] + (-1.0/12)*GRID[p+2,g,t,d])*Nm[p,g,t,s,d,0] / dphi
#                          elif p == 1:
#                              dG_dp_oth = (                           (-2.0/3)*GRID[p-1,g,t,d] + (2.0/3)*GRID[p+1,g,t,d] + (-1.0/12)*GRID[p+2,g,t,d])*Nm[p,g,t,s,d,0] / dphi
#                          elif p == phiL-2:
#                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d] + (-2.0/3)*GRID[p-1,g,t,d] + (2.0/3)*GRID[p+1,g,t,d]                            )*Nm[p,g,t,s,d,0] / dphi
#                          elif p == phiL-1:
#                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d] + (-2.0/3)*GRID[p-1,g,t,d]                                                      )*Nm[p,g,t,s,d,0] / dphi
#                          else:
#                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d] + (-2.0/3)*GRID[p-1,g,t,d] + (2.0/3)*GRID[p+1,g,t,d] + (-1.0/12)*GRID[p+2,g,t,d])*Nm[p,g,t,s,d,0] / dphi
#                          # gamma
#                          if g == 0:
#                              dG_dg_oth = (                                                      (2.0/3)*GRID[p,g+1,t,d] + (-1.0/12)*GRID[p,g+2,t,d])*Nm[p,g,t,s,d,1] / dgam
#                          elif g == 1:
#                              dG_dg_oth = (                           (-2.0/3)*GRID[p,g-1,t,d] + (2.0/3)*GRID[p,g+1,t,d] + (-1.0/12)*GRID[p,g+2,t,d])*Nm[p,g,t,s,d,1] / dgam
#                          elif g == gamL-2:
#                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d] + (-2.0/3)*GRID[p,g-1,t,d] + (2.0/3)*GRID[p,g+1,t,d]                            )*Nm[p,g,t,s,d,1] / dgam
#                          elif g == gamL-1:
#                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d] + (-2.0/3)*GRID[p,g-1,t,d]                                                      )*Nm[p,g,t,s,d,1] / dgam
#                          else:
#                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d] + (-2.0/3)*GRID[p,g-1,t,d] + (2.0/3)*GRID[p,g+1,t,d] + (-1.0/12)*GRID[p,g+2,t,d])*Nm[p,g,t,s,d,1] / dgam
#                          # theta
#                          if t == 0:
#                              dG_dt_oth = (                                                      (2.0/3)*GRID[p,g,t+1,d] + (-1.0/12)*GRID[p,g,t+2,d])*Nm[p,g,t,s,d,2] / dthe
#                          elif t == 1:
#                              dG_dt_oth = (                           (-2.0/3)*GRID[p,g,t-1,d] + (2.0/3)*GRID[p,g,t+1,d] + (-1.0/12)*GRID[p,g,t+2,d])*Nm[p,g,t,s,d,2] / dthe
#                          elif t == theL-2:
#                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d] + (-2.0/3)*GRID[p,g,t-1,d] + (2.0/3)*GRID[p,g,t+1,d]                            )*Nm[p,g,t,s,d,2] / dthe
#                          elif t == theL-1:
#                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d] + (-2.0/3)*GRID[p,g,t-1,d]                                                      )*Nm[p,g,t,s,d,2] / dthe
#                          else:
#                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d] + (-2.0/3)*GRID[p,g,t-1,d] + (2.0/3)*GRID[p,g,t+1,d] + (-1.0/12)*GRID[p,g,t+2,d])*Nm[p,g,t,s,d,2] / dthe
#                       elif s > d:
#                          # phi
#                          if p == 0:
#                              dG_dp_oth = (                                                                                          (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0] + (-1.0/12)*GRID[p+2,g,t,d]*Nm[p+2,g,t,s,d,0]) / dphi
#                          elif p == 1:
#                              dG_dp_oth = (                                             (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0] + (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0] + (-1.0/12)*GRID[p+2,g,t,d]*Nm[p+2,g,t,s,d,0]) / dphi
#                          elif p == phiL-2:
#                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d]*Nm[p-2,g,t,s,d,0] + (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0] + (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0]                                              ) / dphi
#                          elif p == phiL-1:
#                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d]*Nm[p-2,g,t,s,d,0] + (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0]                                                                                          ) / dphi
#                          else:
#                              dG_dp_oth = ((1.0/12)*GRID[p-2,g,t,d]*Nm[p-2,g,t,s,d,0] + (-2.0/3)*GRID[p-1,g,t,d]*Nm[p-1,g,t,s,d,0] + (2.0/3)*GRID[p+1,g,t,d]*Nm[p+1,g,t,s,d,0] + (-1.0/12)*GRID[p+2,g,t,d]*Nm[p+2,g,t,s,d,0]) / dphi
#                          # gam
#                          if g == 0:
#                              dG_dg_oth = (                                                                                          (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1] + (-1.0/12)*GRID[p,g+2,t,d]*Nm[p,g+2,t,s,d,1]) / dgam
#                          elif g == 1:
#                              dG_dg_oth = (                                             (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1] + (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1] + (-1.0/12)*GRID[p,g+2,t,d]*Nm[p,g+2,t,s,d,1]) / dgam
#                          elif g == gamL-2:
#                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d]*Nm[p,g-2,t,s,d,1] + (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1] + (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1]                                              ) / dgam
#                          elif g == gamL-1:
#                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d]*Nm[p,g-2,t,s,d,1] + (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1]                                                                                          ) / dgam
#                          else:
#                              dG_dg_oth = ((1.0/12)*GRID[p,g-2,t,d]*Nm[p,g-2,t,s,d,1] + (-2.0/3)*GRID[p,g-1,t,d]*Nm[p,g-1,t,s,d,1] + (2.0/3)*GRID[p,g+1,t,d]*Nm[p,g+1,t,s,d,1] + (-1.0/12)*GRID[p,g+2,t,d]*Nm[p,g+2,t,s,d,1]) / dgam
#                          # the
#                          if t == 0:
#                              dG_dt_oth = (                                                                                          (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2] + (-1.0/12)*GRID[p,g,t+2,d]*Nm[p,g,t+2,s,d,2]) / dthe
#                          elif t == 1:
#                              dG_dt_oth = (                                             (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2] + (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2] + (-1.0/12)*GRID[p,g,t+2,d]*Nm[p,g,t+2,s,d,2]) / dthe
#                          elif t == theL-2:
#                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d]*Nm[p,g,t-2,s,d,2] + (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2] + (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2]                                              ) / dthe
#                          elif t == theL-1:
#                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d]*Nm[p,g,t-2,s,d,2] + (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2]                                                                                          ) / dthe
#                          else:
#                              dG_dt_oth = ((1.0/12)*GRID[p,g,t-2,d]*Nm[p,g,t-2,s,d,2] + (-2.0/3)*GRID[p,g,t-1,d]*Nm[p,g,t-1,s,d,2] + (2.0/3)*GRID[p,g,t+1,d]*Nm[p,g,t+1,s,d,2] + (-1.0/12)*GRID[p,g,t+2,d]*Nm[p,g,t+2,s,d,2]) / dthe
#                       else:
#                          dG_dp_oth = 0
#                          dG_dg_oth = 0
#                          dG_dt_oth = 0
#
#                       Ntot = Ntot - (dG_dp_oth + dG_dg_oth + dG_dt_oth)
#
#                   if selector == 1:
#                       #new[p,g,t,s] = I * (Ttot + Vtot + Mtot + Ntot)
#                       new[p,g,t,s] = I * (Ttot + Vtot + Atot + Mtot + Ntot)
#                   else:
#                       kinS[p,g,t,s] = Ttot + Ntot
#                       potS[p,g,t,s] = Vtot
#                       pulS[p,g,t,s] = Mtot
#                       absS[p,g,t,s] = Atot
#
#    if selector == 1:
#        return(new)
#    else:
#        return(kinS,potS,pulS,absS)


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
    NEEDS TO BE CORRECTED
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
    NEEDS TO BE CORRECTED
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
    NEEDS TO BE CORRECTED
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
        double dthe=inp['dthe'],V,Ab
        double [:,:] Vm = inp['potCube']
        double [:,:] absorb = inp['absorb']
        double [:,:,:] Km = inp['kinCube']
        double [:,:,:,:] Dm = inp['dipCube']
        double [:,:,:,:] Nm = inp['nacCube']
        double [:] pulseV
        double complex [:,:] new, kinS, potS, pulS, absS
        double complex I = -1j
        double complex dG_dt, d2G_dt2, G, dG_dt_oth
        double complex Ttt
        double complex Ttot,Vtot,Mtot,Ntot,Atot

    new = np.empty_like(GRID)
    kinS = np.empty_like(GRID)
    potS = np.empty_like(GRID)
    pulS = np.empty_like(GRID)
    absS = np.empty_like(GRID)

    pulseV = np.empty((3))

    pulseV[0] = pulZe(time,inp['pulseX'])
    pulseV[1] = pulZe(time,inp['pulseY'])
    pulseV[2] = pulZe(time,inp['pulseZ'])

    for s in range(nstates):
        for t in range(theL):
            G = GRID[t,s]
            V = Vm[t,s]
            Ab = absorb[t,s]

            # derivatives in theta
            if t == 0:
                dG_dt   = ((2.0/3)*GRID[t+1,s]+(-1.0/12)*GRID[t+2,s]) / dthe
                d2G_dt2 = (-GRID[t+2,s]+16*GRID[t+1,s]-30*GRID[t,s]) / (12 * dthe**2)

            elif t == 1:
                dG_dt   = ((-2.0/3)*GRID[t-1,s]+(2.0/3)*GRID[t+1,s]+(-1.0/12)*GRID[t+2,s]) / dthe
                d2G_dt2 = (-GRID[t+2,s]+16*GRID[t+1,s]-30*GRID[t,s]+16*GRID[t-1,s]) / (12 * dthe**2)

            elif t == theL-2:
                dG_dt   = ((1.0/12)*GRID[t-2,s]+(-2.0/3)*GRID[t-1,s]+(2.0/3)*GRID[t+1,s]) / dthe
                d2G_dt2 = (+16*GRID[t+1,s]-30*GRID[t,s]+16*GRID[t-1,s]-GRID[t-2,s]) / (12 * dthe**2)

            elif t == theL-1:
                dG_dt   = ((1.0/12)*GRID[t-2,s]+(-2.0/3)*GRID[t-1,s]) / dthe
                d2G_dt2 = (-30*GRID[t,s]+16*GRID[t-1,s]-GRID[t-2,s]) / (12 * dthe**2)

            else:
                dG_dt   = ((1.0/12)*GRID[t-2,s]+(-2.0/3)*GRID[t-1,s]+(2.0/3)*GRID[t+1,s]+(-1.0/12)*GRID[t+2,s]) / dthe
                d2G_dt2 = (-GRID[t+2,s]+16*GRID[t+1,s]-30*GRID[t,s]+16*GRID[t-1,s]-GRID[t-2,s]) / (12 * dthe**2)


            # T elements (9)
            Ttt = Km[t,8,0] * G + Km[t,8,1] * dG_dt + Km[t,8,2] * d2G_dt2
            #Ttt = -0.0000001 * d2G_dt2

            Ttot = Ttt
            Vtot = V * G
            Atot = - Ab * G * 1j

            # loop and sum on other states.
            Mtot = 0
            Ntot = 0

            for d in range(nstates): # state s is where the outer loop is, d is where the inner loop is.
                for carte in range(3): # carte is 'cartesian', meaning 0,1,2 -> x,y,z
                    Mtot = Mtot - ((pulseV[carte] * Dm[t,carte,s,d] ) * GRID[t,d])

                # NAC calculation
                if   s < d:
                   if t == 0:
                       dG_dt_oth = (                                          (2.0/3)*GRID[t+1,d] + (-1.0/12)*GRID[t+2,d])*Nm[t,s,d,2] / dthe
                   elif t == 1:
                       dG_dt_oth = (                     (-2.0/3)*GRID[t-1,d] + (2.0/3)*GRID[t+1,d] + (-1.0/12)*GRID[t+2,d])*Nm[t,s,d,2] / dthe
                   elif t == theL-2:
                       dG_dt_oth = ((1.0/12)*GRID[t-2,d] + (-2.0/3)*GRID[t-1,d] + (2.0/3)*GRID[t+1,d]                      )*Nm[t,s,d,2] / dthe
                   elif t == theL-1:
                       dG_dt_oth = ((1.0/12)*GRID[t-2,d] + (-2.0/3)*GRID[t-1,d]                                          )*Nm[t,s,d,2] / dthe
                   else:
                       dG_dt_oth = ((1.0/12)*GRID[t-2,d] + (-2.0/3)*GRID[t-1,d] + (2.0/3)*GRID[t+1,d] + (-1.0/12)*GRID[t+2,d])*Nm[t,s,d,2] / dthe
                elif s > d:
                   if t == 0:
                       dG_dt_oth = (                                                                      (2.0/3)*GRID[t+1,d]*Nm[t+1,s,d,2] + (-1.0/12)*GRID[t+2,d]*Nm[t+2,s,d,2]) / dthe
                   elif t == 1:
                       dG_dt_oth = (                                   (-2.0/3)*GRID[t-1,d]*Nm[t-1,s,d,2] + (2.0/3)*GRID[t+1,d]*Nm[t+1,s,d,2] + (-1.0/12)*GRID[t+2,d]*Nm[t+2,s,d,2]) / dthe
                   elif t == theL-2:
                       dG_dt_oth = ((1.0/12)*GRID[t-2,d]*Nm[t-2,s,d,2] + (-2.0/3)*GRID[t-1,d]*Nm[t-1,s,d,2] + (2.0/3)*GRID[t+1,d]*Nm[t+1,s,d,2]                                    ) / dthe
                   elif t == theL-1:
                       dG_dt_oth = ((1.0/12)*GRID[t-2,d]*Nm[t-2,s,d,2] + (-2.0/3)*GRID[t-1,d]*Nm[t-1,s,d,2]                                                                      ) / dthe
                   else:
                       dG_dt_oth = ((1.0/12)*GRID[t-2,d]*Nm[t-2,s,d,2] + (-2.0/3)*GRID[t-1,d]*Nm[t-1,s,d,2] + (2.0/3)*GRID[t+1,d]*Nm[t+1,s,d,2] + (-1.0/12)*GRID[t+2,d]*Nm[t+2,s,d,2]) / dthe
                else:
                     dG_dt_oth = 0


                Ntot = Ntot - dG_dt_oth
                #if lol > 0.00001:
                    #print(time,t,s,d,lol,dG_dt_oth,dG_dt,Nm[t,s,d,2],Nm[t,d,s,2])
                #if selector == 1 and t == 70 and s == 0 and d == 1:
                #    print(time,t,s,d)
                #    print(Nm[t,s,d,2],dthe,2.0/3)
                #    print(((2.0/3)*Nm[t,s,d,2])/dthe)
                #if selector == 1 and t == 70 and s == 1 and d == 0:
                #    print(time,t,s,d)
                #    print(Nm[t-1,s,d,2],dthe,-2.0/3)
                #    print(((-2.0/3)*Nm[t-1,s,d,2])/dthe)
            #print(Ntot)
            if selector == 1:
                new[t,s] = I * (Ttot + Vtot + Mtot + Ntot + Atot)
            else:
                kinS[t,s] = Ttot + Ntot
                potS[t,s] = Vtot
                pulS[t,s] = Mtot
                absS[t,s] = Atot

    if selector == 1:
        return(new)
    else:
        return(kinS,potS,pulS,absS)

