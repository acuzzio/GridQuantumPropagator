import numpy as np
cimport numpy as np
#import quantumpropagator.EMPulse as pp
cimport cython

#import pyximport; pyximport.install()

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

def Cderivative2dGamThe(t,GRID,inp):
    '''wrapper'''
    return np.asarray(Cderivative2dGamTheC(t,GRID,inp))

cdef extern from "complex.h":
        double complex cexp(double complex)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef Cderivative2dGamTheC(double time,double complex [:,:] GRID,dict inp):
    '''
    derivative done for a 2d Grid on the angles
    '''

    cdef:
        int g,t,gamL=inp['gamL'],theL=inp['theL']
        double dgam = inp['dgam'], dthe = inp['dthe'], V
        double complex dG_dg, d2G_dg2, d2G_dgt_numerator_g, d2G_dgt_numerator_t, dG_dt, d2G_dt2
        double complex d2G_dgt_numerator_cross_1, d2G_dgt_numerator_cross_2, d2G_dgt_numerator, G
        double complex d2G_dgt, d2G_dtg, Tgg,Tgt,Ttg,Ttt,Ttot,Vtot
        double [:,:] K
        double [:,:,:] Vm = inp['potCube']
        double [:,:,:,:] Km = inp['kinCube']
        double complex [:,:] new
        double complex I = -1j

    new = np.empty_like(GRID)
    #kintotSum = 0
    #pottotSum = 0
    for g in range(gamL):
       for t in range(theL):
           G = GRID[g,t]
           V = Vm[g,t,0]
           K = Km[g,t]

           # derivatives in gam
           if g == 0:
               dG_dg   = (GRID[g+1,t]) / (2 * dgam)
               d2G_dg2 = (-GRID[g+2,t]+16*GRID[g+1,t]-30*GRID[g,t]) / (12 * dgam**2)
               d2G_dgt_numerator_g = -GRID[g+1,t]

           elif g == 1:
               dG_dg   = (GRID[g+1,t]-GRID[g-1,t]) / (2 * dgam)
               d2G_dg2 = (-GRID[g+2,t]+16*GRID[g+1,t]-30*GRID[g,t]+16*GRID[g-1,t]) / (12 * dgam**2)
               d2G_dgt_numerator_g = -GRID[g+1,t] -GRID[g-1,t]

           elif g == gamL-2:
               dG_dg   = (GRID[g+1,t]-GRID[g-1,t]) / (2 * dgam)
               d2G_dg2 = (+16*GRID[g+1,t]-30*GRID[g,t]+16*GRID[g-1,t]-GRID[g-2,t]) / (12 * dgam**2)
               d2G_dgt_numerator_g = -GRID[g+1,t] -GRID[g-1,t]

           elif g == gamL-1:
               dG_dg   = (-GRID[g-1,t]) / (2 * dgam)
               d2G_dg2 = (-30*GRID[g,t]+16*GRID[g-1,t]-GRID[g-2,t]) / (12 * dgam**2)
               d2G_dgt_numerator_g = -GRID[g-1,t]

           else:
               dG_dg   = (GRID[g+1,t]-GRID[g-1,t]) / (2 * dgam)
               d2G_dg2 = (-GRID[g+2,t]+16*GRID[g+1,t]-30*GRID[g,t]+16*GRID[g-1,t]-GRID[g-2,t]) / (12 * dgam**2)
               d2G_dgt_numerator_g = -GRID[g+1,t] -GRID[g-1,t]

           # derivatives in the
           if t == 0:
               dG_dt   = (GRID[g,t+1]) / (2 * dthe)
               d2G_dt2 = (-GRID[g,t+2]+16*GRID[g,t+1]-30*GRID[g,t]) / (12 * dthe**2)
               d2G_dgt_numerator_t = -GRID[g,t+1]

           elif t == 1:
               dG_dt   = (GRID[g,t+1]-GRID[g,t-1]) / (2 * dthe)
               d2G_dt2 = (-GRID[g,t+2]+16*GRID[g,t+1]-30*GRID[g,t]+16*GRID[g,t-1]) / (12 * dthe**2)
               d2G_dgt_numerator_t = -GRID[g,t+1] -GRID[g,t-1]

           elif t == theL-2:
               dG_dt   = (GRID[g,t+1]-GRID[g,t-1]) / (2 * dthe)
               d2G_dt2 = (+16*GRID[g,t+1]-30*GRID[g,t]+16*GRID[g,t-1]-GRID[g,t-2]) / (12 * dthe**2)
               d2G_dgt_numerator_t = -GRID[g,t+1] -GRID[g,t-1]

           elif t == theL-1:
               dG_dt   = (-GRID[g,t-1]) / (2 * dthe)
               d2G_dt2 = (-30*GRID[g,t]+16*GRID[g,t-1]-GRID[g,t-2]) / (12 * dthe**2)
               d2G_dgt_numerator_t = -GRID[g,t-1]

           else:
               dG_dt   = (GRID[g,t+1]-GRID[g,t-1]) / (2 * dthe)
               d2G_dt2 = (-GRID[g,t+2]+16*GRID[g,t+1]-30*GRID[g,t]+16*GRID[g,t-1]-GRID[g,t-2]) / (12 * dthe**2)
               d2G_dgt_numerator_t = -GRID[g,t+1] -GRID[g,t-1]


           # cross terms: they're 2?
           if g == 0 or t == 0:
               d2G_dgt_numerator_cross_1 = 0
           else:
               d2G_dgt_numerator_cross_1 = +GRID[g-1,t-1]

           if g == gamL-1 or t == theL-1:
               d2G_dgt_numerator_cross_2 = 0
           else:
               d2G_dgt_numerator_cross_2 = +GRID[g+1,t+1]


           d2G_dgt_numerator = d2G_dgt_numerator_g + d2G_dgt_numerator_t + d2G_dgt_numerator_cross_1 + d2G_dgt_numerator_cross_2 + 2*G
           d2G_dgt = d2G_dgt_numerator/(2*dgam*dthe)
           d2G_dtg = d2G_dgt

           # T elements
           #Tgg = K[4,0] * G + K[4,1] * dG_dg + K[4,2] * d2G_dg2
           #Tgt = K[5,0] * G + K[5,1] * dG_dg + K[5,2] * d2G_dgt
           #Ttg = K[7,0] * G + K[7,1] * dG_dt + K[7,2] * d2G_dtg
           #Ttt = K[8,0] * G + K[8,1] * dG_dt + K[8,2] * d2G_dt2
           Tgg =  K[4,2] * d2G_dg2
           Tgt =  K[5,2] * d2G_dgt
           Ttg =  K[7,2] * d2G_dtg
           Ttt =  K[8,2] * d2G_dt2

           Ttot = (Tgg + Tgt + Ttg + Ttt)
           Vtot = V * G
           #kintotSum += Ttot
           #pottotSum += Vtot

           #prr = False
           #if prr == True:
           #    print()
           #    print(K)
           #    print('d1: {:e} {:e}'.format(dG_dg,dG_dt))
           #    print('d2: {:e} {:e}'.format(d2G_dg2,d2G_dt2))
           #    print('T: {:e} {:e} {:e} {:e}'.format(Tgg, Tgt, Ttg, Ttt))
           #    print('({},{})    Ttot: {:.2f}      Vtot: {:.2f}   elem: {:.2f}'.format(g,t,Ttot,Vtot, (-1j * (Ttot+Vtot))))

           new[g,t] = I * (Ttot+Vtot)
    #print('Sum on the grid -> Kin {:e} {:+e} i ###  Pot {:e} {:+e} i'.format(kintotSum.real,kintotSum.imag, pottotSum.real, pottotSum.imag))
    return new

