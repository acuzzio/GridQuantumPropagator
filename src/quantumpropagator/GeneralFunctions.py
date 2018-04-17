'''
This is the module for general purposes functions
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import yaml
import sys

#Debug time
#import pdb
#pdb.set_trace() #to debug h=help

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, bar_length=60):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    total = total -1
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def bring_input_to_AU(iDic):
    '''
    this function is here to make the conversions between fs/ev and AU
    inputDict :: Dict
    '''
    iDic['dt'] = fromFsToAu(iDic['dt'])
    iDic['fullTime'] = fromFsToAu(iDic['fullTime'])
    # change sigmas and T_0s
    iDic['pulseX'][2] = fromFsToAu(iDic['pulseX'][2])
    iDic['pulseX'][4] = fromFsToAu(iDic['pulseX'][4])
    iDic['pulseY'][2] = fromFsToAu(iDic['pulseY'][2])
    iDic['pulseY'][4] = fromFsToAu(iDic['pulseY'][4])
    iDic['pulseZ'][2] = fromFsToAu(iDic['pulseZ'][2])
    iDic['pulseZ'][4] = fromFsToAu(iDic['pulseZ'][4])
    return (iDic)


def readDirectionFile(fn):
    '''
    fn :: filePath
    '''
    with open(fn,'r') as f:
        f.readline()
        phis = f.readline()
        f.readline()
        f.readline()
        gammas = f.readline()
        f.readline()
        f.readline()
        thetas = f.readline()
    return(phis.rstrip().split(' '),gammas.rstrip().split(' '), thetas.rstrip().split(' '))


def printDict(dictionary):
    '''
    pretty printer for dictionary
    dictionary :: Dictionary
    '''
    for x in dictionary:
        print('{} -> {}'.format(x,dictionary[x]))

def printDictKeys(dictionary):
    print(dictionary.keys())

def readGeometry(fn):
    '''
    It gives back the geometry from a file
    fn :: String <- filepath
    '''
    with open(fn,'r') as f:
        data = f.readlines()
    natom = int(data[0])
    title = data[1]
    geomVector = np.empty((natom,3))
    atomType = []
    for i in range(natom):
        atom = list(filter(None, data[i+2].split(' ')))
        atomType.append(atom[0])
        geomVector[i,0] = float(atom[1])
        geomVector[i,1] = float(atom[2])
        geomVector[i,2] = float(atom[3])
    return(natom,title,atomType,geomVector)


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


def chunksOf(xs, n):
    """Yield successive n-sized chunks from xs"""
    shape0 = xs.shape[0]
    for i in range(0, shape0, n):
        yield xs[i:i + n]


def chunksOfList(xs, n):
    """Yield successive n-sized chunks from xs"""
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


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


def gaussian2(x, x0, gw):
    '''
    It calculates the gaussian value at point x. This gaussian is not normalized because
    in this problem the normalization is done at the end.
    x :: Double - the x point
    x0 :: Double - the displacement on the x axis
    gw :: Double - the value of the gw factor in front of the equation
    '''
    return np.exp((- gw * (x - x0)**2) / 2) + 0j


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

def fromFsToAu(n):
    ''' from femtosecond to au '''
    return (n*41.341)

def fromBohToAng(n):
    ''' From Bohr to Angstrom conversion - n :: Double '''
    return (n * 0.529177249)

def fromAngToBoh(n):
    ''' From Angstrom to Bohr conversion - n :: Double '''
    return (n * 1.889725988)

def fromEvtoHar(n):
    ''' From ElectronVolt to Hartree conversion - n :: Double '''
    return (n * 0.0367493)

def fromHartoEv(n):
    ''' From Hartree to ElectronVolt conversion - n :: Double '''
    return (n * 27.211402)

def fromCmMin1toHartree(n):
    ''' from cm-1 to hartree conversion - n :: Double '''
    return (n*4.5563e-06)

def fromHartreetoCmMin1(n):
    ''' from hartree to cm-1 conversion - n :: Double '''
    return (n/4.5563e-06)


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

def calcBond(geom,atom1,atom2):
    '''
    returns the bond length between atom1 and atom2
    geom :: np.array(natoms,3)
    atom1 = integer
    atom2 = integer
    '''
    a = geom[atom1-1]
    b = geom[atom2-1]
    bond = np.linalg.norm(a-b)
    return bond

def calcAngle(geom,atom1,atom2,atom3):
    '''
    returns the angle between atom1,2 and 3
    geom :: np.array(natoms,3)
    atom1 = integer
    atom2 = integer
    atom3 = integer
    '''
    a = geom[atom1-1]
    b = geom[atom2-1]
    c = geom[atom3-1]
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return(np.degrees(angle))

def calcDihedral(geom,atom1,atom2,atom3,atom4):
    '''
    returns the dihedral of atom1,2,3 and 4
    geom :: np.array(natoms,3)
    atom1 = integer
    atom2 = integer
    atom3 = integer
    atom4 = integer
    '''
    a = geom[atom1-1]
    b = geom[atom2-1]
    c = geom[atom3-1]
    d = geom[atom4-1]
    print('still have to do it')
    print(a,b,c,d)

def massOf(elem):
    '''
    You get the mass of an element from the label string
    elem :: String
    '''
    dictMass = {'X': 0, 'Ac': 227.028, 'Al': 26.981539, 'Am': 243, 'Sb': 121.757, 'Ar':
            39.948, 'As': 74.92159, 'At': 210, 'Ba': 137.327, 'Bk': 247, 'Be':
            9.012182, 'Bi': 208.98037, 'Bh': 262, 'B': 10.811, 'Br': 79.904,
            'Cd': 112.411, 'Ca': 40.078, 'Cf': 251, 'C': 12.011, 'Ce': 140.115,
            'Cs': 132.90543, 'Cl': 35.4527, 'Cr': 51.9961, 'Co': 58.9332, 'Cu':
            63.546, 'Cm': 247, 'Db': 262, 'Dy': 162.5, 'Es': 252, 'Er': 167.26,
            'Eu': 151.965, 'Fm': 257, 'F': 18.9984032, 'Fr': 223, 'Gd': 157.25,
            'Ga': 69.723, 'Ge': 72.61, 'Au': 196.96654, 'Hf': 178.49, 'Hs':
            265, 'He': 4.002602, 'Ho': 164.93032, 'H': 1.00794, 'In': 114.82,
            'I': 126.90447, 'Ir': 192.22, 'Fe': 55.847, 'Kr': 83.8, 'La':
            138.9055, 'Lr': 262, 'Pb': 207.2, 'Li': 6.941, 'Lu': 174.967, 'Mg':
            24.305, 'Mn': 54.93805, 'Mt': 266, 'Md': 258, 'Hg': 200.59, 'Mo': 95.94,
            'Nd': 144.24, 'Ne': 20.1797, 'Np': 237.048, 'Ni': 58.6934, 'Nb': 92.90638,
            'N': 14.00674, 'No': 259, 'Os': 190.2, 'O': 15.9994, 'Pd': 106.42, 'P':
            30.973762, 'Pt': 195.08, 'Pu': 244, 'Po': 209, 'K': 39.0983, 'Pr':
            140.90765, 'Pm': 145, 'Pa': 231.0359, 'Ra': 226.025, 'Rn': 222,
            'Re': 186.207, 'Rh': 102.9055, 'Rb': 85.4678, 'Ru': 101.07, 'Rf':
            261, 'Sm': 150.36, 'Sc': 44.95591, 'Sg': 263, 'Se': 78.96, 'Si':
            28.0855, 'Ag': 107.8682, 'Na': 22.989768, 'Sr': 87.62, 'S': 32.066,
            'Ta': 180.9479, 'Tc': 98, 'Te': 127.6, 'Tb': 158.92534, 'Tl':
            204.3833, 'Th': 232.0381, 'Tm': 168.93421, 'Sn': 118.71, 'Ti':
            47.88, 'W': 183.85, 'U': 238.0289, 'V': 50.9415, 'Xe': 131.29,
            'Yb': 173.04, 'Y': 88.90585, 'Zn': 65.39, 'Zr': 91.224}
    return(dictMass[elem])

def saveTraj(arrayTraj, labels, filename, convert=None):
    '''
    given a numpy array of multiple coordinates, it prints the concatenated xyz file
    arrayTraj :: np.array(ncoord,natom,3)    <- the coordinates
    labels :: [String] <- ['C', 'H', 'Cl']
    filename :: String <- filepath
    convert :: Bool <- it tells if you need to convert from Boh to Ang (default True)
    '''
    convert = convert or False
    (ncoord,natom,_) = arrayTraj.shape
    fn = filename + '.xyz'
    string = ''
    for geo in range(ncoord):
        string += str(natom) + '\n\n'
        for i in range(natom):
            if convert:
                string += "   ".join([labels[i]] +
                        ['{:10.6f}'.format(fromBohToAng(num)) for num
                    in arrayTraj[geo,i]]) + '\n'
            else:
                string += "   ".join([labels[i]] +
                        ['{:10.6f}'.format(num) for num
                    in arrayTraj[geo,i]]) + '\n'


    with open(fn, "w") as myfile:
        myfile.write(string)
    print('\nfile {0} written:\n\nvmd {0}'.format(fn))

def scanvalues(first,second,resolution):
    '''
    This uses numpy to get the values printed out in a single line.
    first  :: Double <- start of the interval
    second :: Double <- end of the interval
    resolution :: Int <- resolution (how many points in the interval
    '''
    vec = np.linspace(first,second,resolution)
    oneline = " ".join(['{:7.3f}'.format(b) for b in vec])
    return oneline

def printMatrix2D(mat, pre=None, thr=None):
    '''
    mat :: np.array(X,Y) <- I use this for overlap matrix
    pre :: Int  <- the precision for the output
    thr :: Double <- value smaller than this are set to 0
    given a 2d array in numpy, it prints the matrix on the screen
    '''
    pre = pre or 6
    thr = thr or 0.0
    pd.set_option('precision', pre)
    pd.set_option('chop_threshold', thr)
    (siza,_) = mat.shape
    indexes = np.arange(siza) + 1
    out = pd.DataFrame(mat, index=indexes, columns=indexes)
    print(out)

def createTabellineFromArray(arr):
    '''
    arr :: np.array(Double)
    This function will take a 1D numpy array and create a matrix with
    the element multiplications (the product between the cartesian product)
    '''
    length = arr.size
    mat = np.empty((length,length))
    for ii in np.arange(length):
        for kk in np.arange(length):
            mat[ii,kk]=arr[ii]*arr[kk]
    return(mat)

def labTranform(string):
    '''
    transform the string of the form
    P014-800
    into his +14.8 float
    '''
    return (float(string.replace('-','.').replace('N','-').replace('P','+')))


def labTranformA(strings):
    '''
    transform an array of string of the form
    P014-800
    into his +14.8 float type numpy array : D
    '''
    return (np.array([labTranform(a) for a in strings]))


def stringTransformation3d(fn):
    '''
    transform the string of the form
    'h5/zNorbornadiene_N006-400_P014-800_P085-500.rassi.h5'
    into 3 numbers and 3 labels
    '''
    fn1 = fn.split('.')[0]  # h5/zNorbornadiene_N006-400_P014-800_P085-500
    # str1 = 'N006-400' ->  axis1 = -6.4
    [str1,str2,str3] = fn1.split('_')[1:]
    [axis1,axis2,axis3] = [
            labTranform(x) for x in
            [str1,str2,str3]]
    return(axis1,str1,axis2,str2,axis3,str3)


def loadInputYAML(fn):
    '''
    this function reads the input file and returns a dictionary with inputs
    fn :: filePath
    '''
    with open(fn, 'r') as f:
         diction = yaml.safe_load(f)
    return diction


if __name__ == "__main__":
    from time import sleep

    # A List of Items
    items = list(range(0, 57))
    l = len(items)

    # Initial call to print 0% progress
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    for i, item in enumerate(items):
        # Do stuff...
        sleep(0.1)
        # Update Progress Bar
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)



