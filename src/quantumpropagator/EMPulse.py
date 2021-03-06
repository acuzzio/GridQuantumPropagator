'''
This is the module that deals with the electromagnetic pulse
'''

import numpy as np

##################################################
#                                                #
#                 Pulse Section                  #
#                                                #

def specificPulse(t):
    '''
    A defined pulse with the three components x, y and z
    $\color{violet}\vec{E}(t) = \sum_d E_d cos(\omega t + \phi) \cdot e^{-\dfrac{(t-t_0)^2}{2\sigma^2}} \ \ \ \ \ \   d={x,y,z}$
    '''
    pulseX=pulse(t,0.024,0.24,30,0,100)
    pulseY=pulse(t,0.024,0.24,30,0,100)
    pulseZ=pulse(t,0.024,0.24,30,0,100)
    return np.array([pulseX,pulseY,pulseZ],dtype=float) ## AU

def userPulse(t,args):
    '''
    Same pulses as function specificPulse, but not harcoded
    '''
    [[Edx,omegax,sigmax,phix,t0x],[Edy,omegay,sigmay,phiy,t0y],[Edz,omegaz,sigmaz,phiz,t0z]]=args
    pulseX=pulse(t,Edx,omegax,sigmax,phix,t0x)
    pulseY=pulse(t,Edy,omegay,sigmay,phiy,t0y)
    pulseZ=pulse(t,Edz,omegaz,sigmaz,phiz,t0z)
    return np.array([pulseX,pulseY,pulseZ],dtype=float) ## AU
    #return np.array([pulseX,pulseY,pulseZ],dtype=float) ## AU

def varPulseZ(t,Ed,omega,sigma,phi,t0):
    return np.array([0,0,pulse(t,Ed,omega,sigma,phi,t0)],dtype=float) ## AU

def specificPulseZero(t):
    return np.array([0, 0, 0],dtype=float)

def component(t):
    return pulse(t,0.014,0.13,30,0,100)

def envel(t,Ed,sigma,t0):
    '''
    It returns the value of the gaussian envelope for the pulse at time t
    now it works with arrays, too
    '''
    num = (t-t0)**2
    den = 2*(sigma**2)
    if (den == 0):
        if type(t) == float:
            result = 0.0
        else:
            result = np.zeros_like(t)
    else:
        result = Ed * np.exp(-num/den)
    return result

def pulse(t,Ed,omega,sigma,phi,t0):
    '''
    The pulse is a cosine multiplied by a gaussian
    Ed = the wave amplitude along a component :: Double
    omega = the wave length :: Double
    sigma = the width of the gaussian envelope :: Double
    phi = the initial phase :: Double
    t0 = initial value :: Double
    t = the actual time :: Double
    '''
    num = (t-t0)**2
    den = 2*(sigma**2)
    if (den == 0):
        result = 0.0
    else:
        result = Ed * (np.cos(omega*(t-t0)+phi)) * np.exp(-num/den)
    return result


#                                                #
#                                                #
##################################################

if __name__ == "__main__":
    a = userPulse(93,[0.024,0.24,30,0,100])
    print(a)


