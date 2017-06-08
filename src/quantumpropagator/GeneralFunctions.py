import numpy as np
import numba

import multiprocessing as mp

def asyncFun(f,*args,**kwargs):
    job = mp.Process(target=f,args=args,kwargs=kwargs)
    job.start()

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
        return x.real**2 + x.imag**2

#Debug time
#import pdb
#pdb.set_trace() #to debug h=help

def population(grid):
    ''' given a single state it calculates the populations '''
    (nstates,gridN) = grid.shape
    pop = np.apply_along_axis(singlepop,1,grid)
    return(pop,sum(pop))

def ndprint(a, format_string ='{0:15.12f}'):
    return " ".join([format_string.format(v,i) for i,v in enumerate(a)])

def singlepop(GridSingleState):
    return sum(np.apply_along_axis(abs2,0,GridSingleState))

def groundState(n):
    '''
    given the number of states, this creates an array:
    [1,0,0,0,0, ... , 0]
    '''
    a = np.zeros(n)
    a[0]=1.0
    return a

def gaussian(x, mu, sig):
    return (np.exp(-np.power((x - mu)/sig, 2.)/2)) + (0j)

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def saveComplex(fn,array):
    """ Saves a complex array into a txt file """
    # in one column
    #np.savetxt(fn, array.view(float))
    np.savetxt(fn, array.view(float).reshape(-1, 2))

def loadComplex(fn):
    """ Load a complex array from a txt file """
    # in one column
    #array = np.loadtxt('outfile.txt').view(complex)
    return np.loadtxt(fn).view(complex).reshape(-1)

def print2ArrayInColumns(array1,array2,filename):
    """ Saves 2 arrays into 2 columns of a file"""
    np.savetxt(filename,np.stack((array1,array2),1))

def dipoleMoment(states,matMu):
    '''
    dipole moment calculation
    '''
    nstates = states.size
    dipole  = np.zeros(3, dtype = complex)
    for component in [0,1,2]:
        summa = 0
        for Ici in range(nstates):
            for Icj in range(nstates):
                a = np.conjugate(states[Ici])
                b = states[Icj]
                c = matMu[component, Ici, Icj]
                summa += (a*b*c)
                #summa = summa + (a*b*c)
        dipole[component] = summa
    return dipole

def BohToAng(n):
    return (n * 0.529177249)

def EvtoHar(n):
    return (n * 0.0367493)

# https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
# https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def fromCmMin1toHartree(x):
    return (x*4.5563e-06)

