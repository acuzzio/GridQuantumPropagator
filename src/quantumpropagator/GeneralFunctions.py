'''
This is the module for general purposes functions
'''

import multiprocessing as mp
import numpy as np

#Debug time
#import pdb
#pdb.set_trace() #to debug h=help


def calculateGradientOnMatrix0(newNAC,dist):
    '''
    This calculate a matrix gradient along axis 0
    newNAC :: np.array[Double,Double] - derivative coupling matrix
    dist :: np.array[Double] - x values
    '''
    deltaX = dist[1] - dist[0]
    allM = np.apply_along_axis(np.gradient, 0, newNAC, deltaX)
    return allM


def asyncFun(f, *args, **kwargs):
    '''
    Executes the f function on another thread.
    '''
    job = mp.Process(target=f, args=args, kwargs=kwargs)
    job.start()


def abs2(x):
    '''
    x :: complex
    This is a reimplementation of the abs value for complex numbers
    '''
    return x.real**2 + x.imag**2


def population(grid):
    '''
    grid :: np.array[Complex]
    it calculates the populations of a 1D grid
    '''
    pop = np.apply_along_axis(singlepop,1,grid)
    return(pop,sum(pop))


def ndprint(a, format_string ='{0:15.12f}'):
    '''
    a = [Double] :: list of doubles
    It returns a single string of formatted numbers out of a list of doubles
    '''
    return " ".join([format_string.format(v,i) for i,v in enumerate(a)])


def singlepop(GridSingleState):
    '''
    Calculates the population of a single state grid (1D)
    '''
    return sum(np.apply_along_axis(abs2,0,GridSingleState))


def groundState(n):
    '''
    n :: Int
    given the number of states, this creates an array:
    [1,0,0,0,0, ... , 0]
    '''
    a = np.zeros(n)
    a[0]=1.0
    return a


def gaussian(x, mu, sig):
    '''
    It calculates the gaussian value at point x. This gaussian is not normalized because
    in this problem the normalization is done at the end.
    x :: Double - the x point
    mu :: Double - the displacement on the x axis
    sig :: Double - the sigma value
    '''
    return (np.exp(-np.power((x - mu)/sig, 2.)/2)) + (0j)


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
    dipole = np.zeros(3, dtype = complex)
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
    ''' From bohr to Angstrom conversion '''
    return (n * 0.529177249)


def EvtoHar(n):
    ''' From ElectronVolt to Hartree conversion '''
    return (n * 0.0367493)


def fromCmMin1toHartree(x):
    ''' from cm-1 to hartree conversion '''
    return (x*4.5563e-06)


# https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
# https://betterexplained.com/articles/an-interactive-guide-to-the-fourier-transform/
def DFT_slow(x):
    """ Compute the discrete Fourier Transform of the 1D array x """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

