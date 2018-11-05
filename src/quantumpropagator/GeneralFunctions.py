'''
This is the module for general purposes functions
'''

import multiprocessing as mp
import numpy as np
import pandas as pd
import yaml
import sys
import math
import pickle

#Debug time
#import pdb
#pdb.set_trace() #to debug h=help


def equilibriumIndex(fn,dataDict):
    '''
    given the path of direction file and the dataDict, it gives back the index of equilibrium
    points in the array
    fn :: String -> filePath
    dataDict :: {}
    '''
    phis,gams,thes = readDirectionFile(fn)
    gsm_phi_ind = dataDict['phis'].index(phis[0])
    gsm_gam_ind = dataDict['gams'].index(gams[0])
    gsm_the_ind = dataDict['thes'].index(thes[0])
    print('Equilibrium points found at : ({},{},{})'.format(gsm_phi_ind, gsm_gam_ind, gsm_the_ind))
    return (gsm_phi_ind, gsm_gam_ind, gsm_the_ind)


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
    # phi are invariate
    axis1 = axis1/100
    # gamma are converted to radians
    axis2 = np.deg2rad(axis2)
    # theta are divided by 2 and converted to radians
    axis3 = np.deg2rad(axis3/2)
    return(axis1,str1,axis2,str2,axis3,str3)


def fromLabelsToFloats(dataDict):
    '''
    takes the datadict and returns the three arrays of coordinates values
    '''
    phis = labTranformA(dataDict['phis'])/100
    gams = np.deg2rad(labTranformA(dataDict['gams']))
    thes = np.deg2rad(labTranformA(dataDict['thes'])/2)
    return(phis,gams,thes)


def fromFloatsToLabels(phis,gams,thes):
    '''
    it does the opposite of fromLabelsToFloats
    phis,gams,thes :: tuple of three np.array(floats)
    '''
    phiStrings = labTranformReverseA(phis*100)
    gamStrings = labTranformReverseA(np.rad2deg(gams))
    theStrings = labTranformReverseA(np.rad2deg(thes*2))
    return phiStrings, gamStrings, theStrings


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
    bar = '*' * filled_length + '-' * (bar_length - filled_length)

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
    #iDic['pulseX'][2] = fromFsToAu(iDic['pulseX'][2])
    #iDic['pulseX'][4] = fromFsToAu(iDic['pulseX'][4])
    #iDic['pulseY'][2] = fromFsToAu(iDic['pulseY'][2])
    #iDic['pulseY'][4] = fromFsToAu(iDic['pulseY'][4])
    #iDic['pulseZ'][2] = fromFsToAu(iDic['pulseZ'][2])
    #iDic['pulseZ'][4] = fromFsToAu(iDic['pulseZ'][4])
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


def gaussian2(x, x0, gw, moment=None):
    '''
    It calculates the gaussian value at point x. This gaussian is not normalized because
    in this problem the normalization is done at the end.
    x :: Double - the x point
    x0 :: Double - the displacement on the x axis
    gw :: Double - the value of the gw factor in front of the equation
    moment :: Double - the initial moment given to the WF
    '''
    moment = moment or 0
    return np.exp((- gw * (x - x0)**2) / 2) * np.exp(1j*moment*(x - x0))


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


def fromEvtoHart(n):
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


def fromCmMin1toFs(n):
    ''' from cm-1 to fs conversion - n :: Double '''
    return (1/(fromHartreetoCmMin1(n))*1.88365157e+4)


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


def labTranformReverseA(floArray):
    '''
    labTranformReverse applied to an array
    '''
    return [ labTranformReverse(x) for x in floArray ]


def labTranformReverse(flo):
    '''
    from the float number to the labeling of files used in this project
    '''
    flo2 = '{:+08.3f}'.format(flo)
    return flo2.replace('-','N').replace('.','-').replace('+','P')


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


def loadInputYAML(fn):
    '''
    this function reads the input file and returns a dictionary with inputs
    fn :: filePath
    '''
    with open(fn, 'r') as f:
         diction = yaml.safe_load(f)
    return diction


def generateNorbGeometry(phi,gam,the, vector_res=None):
    '''
    This function generates an xyz given the value of the three angles
    phi,gam,the :: Double  <- the three angles
    vector_res :: Boolean <- if false it saves file
    '''
    vector_res = vector_res or False
    fnO = 'zNorbornadiene_{:+08.3f}_{:+08.3f}_{:+08.3f}'.format(phi,gam,the)
    fn = fnO.replace('-','N').replace('.','-').replace('+','P')
    atomT = ['C','C','C','H','H','H','H']
    fixed = np.array([[0.000000, 0.000000, 1.078168],
             [0.000000, -1.116359, 0.000000],
             [0.000000, 1.116359, 0.000000],
             [0.894773, 0.000000, 1.698894],
             [-0.894773, 0.000000, 1.698894],
             [0.000000, -2.148889, 0.336566],
             [0.000000, 2.148889, 0.336566]])
    rBond = 1.541 # distance of bridge
    L = 1.116359  # half distance between C2-C3
    chBond = 1.077194 # distance between moving C and H

    the2 = np.deg2rad(the/2)
    gam2 = np.deg2rad(gam)

    torsionalCI = 6 # values for phi AT WHICH

    # this is the vector that displaces our 8 moving atoms from the CLOSEST CI I
    # can reach with the old scan and the real conical intersection

    deltasCIN = np.array([
                          [-0.165777,  0.067387,  0.016393],
                          [-0.14517 , -0.096085, -0.143594],
                          [ 0.165162, -0.067684,  0.015809],
                          [ 0.145943,  0.095734, -0.143995],
                          [-0.520977,  0.086124,  0.316644],
                          [ 0.450303, -0.048   ,  0.245432],
                          [ 0.520405, -0.086941,  0.316594],
                          [-0.451602,  0.047331,  0.24554 ],
])


    xC1 = -rBond * np.cos(gam2) * np.sin(the2)
    yC1 = L + rBond * - np.sin(gam2)
    zC1 = -rBond * np.cos(the2) * np.cos(gam2)

    xC2 = -rBond * np.cos(gam2) * np.sin(-the2)
    yC2 = L - rBond * np.sin(gam2)
    zC2 = -rBond * np.cos(-the2) * np.cos(gam2)

    xC3 = rBond * np.cos(gam2) * np.sin(+the2)
    yC3 = -L + rBond * np.sin(gam2)
    zC3 = -rBond * np.cos(+the2) * np.cos(gam2)

    xC4 = rBond * np.cos(gam2) * np.sin(-the2)
    yC4 = -L + rBond * np.sin(gam2)
    zC4 = -rBond * np.cos(-the2) * np.cos(gam2)

    # in the end we did this with cartesian... interesting workaround...
    # desperation?
    dx = +0.694921
    dy = +0.661700
    dz = +0.494206

    xH1 = xC1 - dx
    yH1 = yC1 + dy
    zH1 = zC1 - dz

    xH2 = xC2 + dx
    yH2 = yC2 + dy
    zH2 = zC2 - dz

    xH3 = xC3 + dx
    yH3 = yC3 - dy
    zH3 = zC3 - dz

    xH4 = xC4 - dx
    yH4 = yC4 - dy
    zH4 = zC4 - dz


    newAtoms = np.array([[xC1,yC1,zC1], [xC2,yC2,zC2], [xC3,yC3,zC3], [xC4,yC4,zC4],
                [xH1,yH1,zH1], [xH2,yH2,zH2], [xH3,yH3,zH3], [xH4,yH4,zH4]])

    this = ((phi/torsionalCI) * deltasCIN)
    newCorrectedAtoms = newAtoms + this

    new = np.append(fixed,newCorrectedAtoms,0)
    atomTN = atomT + ['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H']
    if vector_res:
        return(new)
    else:
        # saveTraj works on LIST of geometries, that is why the double list brackets
        saveTraj(np.array([new]),atomTN,fn)


def file_len(fname):
    '''
    gives the number of lines in the file
    fn :: FilePath
    '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def frames_counter(fn):
    '''
    Given a trajectory files, gives back number of frame and number of atoms.
    fn :: FilePath
    '''
    f = open(fn)
    atomN = int(f.readline())
    f.close()
    with open(fn) as f:
        for i, l in enumerate(f):
            pass
    frameN = int((i + 1)/(atomN+2))
    return (atomN,frameN)


def readTrajectory(fn):
    '''
    reads a md.xyz format file and gives back a dictionary with geometries and all the rest
    '''
    atomsN,frameN = frames_counter(fn)
    print('\nAtoms: {}\nFrames: {}\n'.format(atomsN,frameN))
    geom = np.empty((frameN,atomsN,3))
    atomT = []
    with open(fn) as f:
        for i in range(frameN):
            f.readline()
            f.readline()
            for j in range(atomsN):
              a  = f.readline()
              bb = a.split(" ")
              b  = [ x for x in bb if x != '']
              geom[i,j] = [float(b[1]),float(b[3]),float(b[2])]
              if i == 0:
                  atomT.append(b[0])
    final_data = {
                 'geoms'  : geom,
                 'atomsN' : atomsN,
                 'frameN' : frameN,
                 'atomT'  : atomT,
                 }
    return final_data

def bondL():
    bondLengths = {
    'HH' : 0.74,
    'CH' : 1.09,
    'HO' : 0.96,
    'HN' : 1.02,
    'CC' : 1.54,
    'CN' : 1.47,
    'CO' : 1.43,
    'NN' : 1.45,
    'NO' : 1.40,
    'OO' : 1.48,
    'HS' : 1.34,
    'OS' : 1.43,
    'CS' : 1.82,
    'NS' : 0.50,
    'SS' : 1.0,
    'II' : 1.0,
    'MM' : 1.0,
    'IM' : 2.8,
    'CM' : 1.0,
    'MS' : 1.0,
    'HM' : 1.0,
    'CI' : 1.0,
    'IS' : 1.0,
    'HI' : 1.0
    }
    return bondLengths


def transformTrajectoryIntoBlenderData(name,traj):
    '''
    takes a trajectory dictionary and gives back new kind of data for blender
    '''
    geoms = traj['geoms']
    atomsN = traj['atomsN']
    frameN = traj['frameN']
    atomT = traj['atomT']
    BL = bondL()
    paletti = []
    spheres = []
    for i in range(atomsN):
        spheres.append((i,atomT[i],geoms[:,i]))
        for j in range(i):
            unoL = atomT[i]
            dueL = atomT[j]
            geom1Ini = geoms[0,i]
            geom2Ini = geoms[0,j]
            toCheckDistance = ''.join(sorted(unoL + dueL))
            bondLengthMax = BL[toCheckDistance] + 0.3
            bondIni = np.linalg.norm((geom2Ini-geom1Ini))
            #print('{} {} {} blMax {}, bondIni {}'.format(i,j,toCheckDistance,bondLengthMax,bondIni))
            if bondIni < bondLengthMax:
                print('There should be a bond between {}{} and {}{}'.format(unoL, i, dueL, j))
                if unoL == dueL:
                    pos1 = geoms[:, i]
                    pos2 = geoms[:, j]
                    paletti.append((pos1,pos2,unoL))
                else:
                    pos1 = geoms[:, i]
                    pos2 = geoms[:, j]
                    center = (pos1 + pos2) / 2
                    paletti.append((pos1,center,unoL))
                    paletti.append((pos2,center,dueL))
    print('{} {}'.format(atomsN,frameN))
    blender_dict = {'spheres' : spheres, 'paletti' : paletti}
    pickle.dump(blender_dict, open(name, "wb" ) )
    print(paletti)
    print('There are {} atoms and {} paletti'.format(len(spheres),len(paletti)))



if __name__ == "__main__":
    fn   = '/home/alessio/Desktop/Dropbox/sharedWithPPl/Acu-Elisa/soloRET/b.xyz'
    name = '/home/alessio/Desktop/Dropbox/sharedWithPPl/Acu-Elisa/soloRET/b.p'
    a = readTrajectory(fn)
    transformTrajectoryIntoBlenderData(name,a)
    #from time import sleep

    ## A List of Items
    #items = list(range(0, 57))
    #l = len(items)

    ## Initial call to print 0% progress
    #printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)
    #for i, item in enumerate(items):
    #    # Do stuff...
    #    sleep(0.1)
    #    # Update Progress Bar
    #    printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', bar_length = 50)



