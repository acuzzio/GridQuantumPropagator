'''
This Script will generate geometries around the conical intersection, linear
and circle.

It needs a geometry file (xyz) and two vectors (that can be read by np.loadtxt)
'''

import numpy as np
import glob
from collections import namedtuple
from argparse import (ArgumentParser,RawTextHelpFormatter)
from quantumpropagator import (readGeometry,saveTraj,err,retrieve_hdf5_data,
                               mathematicaListGenerator, gnuSplotCircle, ndprint)

def read_single_arguments(single_inputs):
    '''
     This funcion reads the command line arguments
    '''
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-v", "--vectors",
                        dest="v",
                        nargs='+',
                        help="the 3 files: geometry and branching plane vectors")
    parser.add_argument("-s", "--scalarProd",
                        dest="s",
                        nargs='+',
                        help="to calculate scalar product between scan \
geometries (given by globalexpression) and branching plane vectors")
    parser.add_argument("-l", "--linear",
                        dest="l",
                        nargs='+',
                        help="parameters for the linear displacement.\n"
                        "Distance :: Double\n"
                        "Number of points :: Int\n")
    parser.add_argument("-c", "--circular",
                        dest="c",
                        nargs='+',
                        help="parameters for the circular displacement.\n"
                        "Number of ponts in the circle :: Int\n"
                        "List of radii :: [Double]")
    parser.add_argument("-g", "--globalPattern",
                        dest="g",
                        type=str,
                        help="it is the global pattern of rasscf h5 files")
    args = parser.parse_args()

    if args.s != None:
        if len(args.s) == 3:
            [globE,grad,der] = args.s
            single_inputs = single_inputs._replace(globExp=globE)
            single_inputs = single_inputs._replace(vectorX=grad)
            single_inputs = single_inputs._replace(vectorY=der)
        else:
            err('this takes 3 arguments: the geometry global expression and the two vectors')
    if args.v != None:
        if len(args.v) == 3:
            [geom,grad,der] = args.v
            single_inputs = single_inputs._replace(fileXYZ=geom)
            single_inputs = single_inputs._replace(vectorX=grad)
            single_inputs = single_inputs._replace(vectorY=der)
        else:
            err('this takes 3 arguments: the single geometry and the two vectors')
    if args.l != None:
        if len(args.l) == 2:
            single_inputs = single_inputs._replace(linearDisplacement=args.l)
        else:
            err('this takes 2 arguments: number of points and distance')
    if args.c != None:
        # without controls because we feel brave
        single_inputs = single_inputs._replace(circleScan=args.c)
    if args.g != None:
        single_inputs = single_inputs._replace(graphsGlob=args.g)
    return single_inputs

def scalarProds(expression,vec1fn,vec2fn):
    '''
    given a scan global expression coordinate and the branching plane vectors,
    it calculates the scalar products
    expression :: String <- the global expression of files
    vec1fn :: String <- filePath
    vec2fn :: String <- filePath
    '''
    GD = np.loadtxt(vec1fn)
    NA = np.loadtxt(vec2fn)
    allfn = sorted(glob.glob(expression))
    (natoms,title,atomTN,_) = readGeometry(allfn[0])
    fileN = len(allfn)
    allgeom = np.empty((fileN, natoms, 3))
    ind = 0
    for f in allfn:
        (_,_,_,geom) = readGeometry(f)
        allgeom[ind] = geom
        ind += 1
    difference = np.diff(allgeom,axis=0)
    GDnorm = np.linalg.norm(GD)
    NAnorm = np.linalg.norm(NA)
    norms = np.linalg.norm(difference, axis =(1,2))
    broadcasted = np.transpose(np.broadcast_to(norms,(3,natoms,fileN-1)))
    unitary_move = difference/broadcasted
    ## I need to divide the vector for the norm
    wow = ndprint(np.tensordot(unitary_move,GD),format_string = '{0:7.4f}')
    wol = ndprint(np.tensordot(unitary_move,NA),format_string = '{0:7.4f}')
    output = '''
    Gd norm -> {}
    Dc norm -> {}

  Scalar product Gd:
{}
  Scalar product Dc:
{}
    '''.format(GDnorm, NAnorm, wow, wol)
    print(output)

def displaceGeom(geom1,xf,yf,linearArgs,circleArgs):
    '''
    This function takes the geometry and the two vectors and creates a circle
    or a line around them. Pretty neat.
    geom1 :: FilePath <- the geometry
    xf :: FilePath <- the gradient difference vector
    yf :: FilePath <- the derivative coupling vector
    linearArgs :: [Int, Double] <- a list of two numbers, the number of points
                                   and the distance.
    circleArgs :: [Int, Double] <- a list with the first number as the number
                                   of points across the circle, and then a list
                                   of any R
    '''
    GD = np.loadtxt(xf)
    NA = np.loadtxt(yf)
    (natoms,title,atomTN,geom) = readGeometry(geom1)
    #print(natoms,title,atomType,geom,GD,NA)
    labelRoot = geom1.split('.')[0]
    if linearArgs != []:
        npoints = int(linearArgs[0])
        distance = float(linearArgs[1])
        for i in np.linspace(-distance,distance,npoints):
            fnO = labelRoot + 'ScanGradDiff_{:+08.3f}'.format(i)
            fn = fnO.replace('-','N').replace('.','-').replace('+','P')
            new = geom + (i*GD)
            saveTraj(np.array([new]),atomTN,fn)

        for i in np.linspace(-distance,distance,npoints):
            fnO = labelRoot + 'ScanDeriCoup_{:+08.3f}'.format(i)
            fn = fnO.replace('-','N').replace('.','-').replace('+','P')
            new = geom + (i*NA)
            saveTraj(np.array([new]),atomTN,fn)

    if circleArgs != []:
        circles = int(circleArgs[0])
        Rlist = [float(a) for a in circleArgs[1:]]
        # this false in linspace function avoids the creation of both 0 and 360 degrees
        for i in np.linspace(0,360,circles,False):
            for R in Rlist:
                fnO = labelRoot + 'ScanRing{:+08.3f}_Angle{:+08.3f}'.format(R,i)
                fn = fnO.replace('-','N').replace('.','-').replace('+','P')
                rad = np.deg2rad(i)
                component1 = R * GD * np.sin(rad)
                component2 = R * NA * np.cos(rad)
                new = geom + component1 + component2
                saveTraj(np.array([new]),atomTN,fn)

def graphScan(globalExp):
    '''
    So, this one takes the global expression of rasscf h5 files and creates the
    circle graph. This assumes that you used the {:+08.3f} convention to create this
    file.
          RingP000-100_AngleP000-000
    glob :: string <- global pattern "*.*"
    '''
    allH5 = sorted(glob.glob(globalExp))
    dime = len(allH5)
    allH5First = allH5[0]
    nstates = len(retrieve_hdf5_data(allH5First,'ROOT_ENERGIES'))
    bigArrayE = np.empty((dime,nstates))
    bigArray1 = np.empty(dime)
    bigArray2 = np.empty(dime)
    ind=0
    for fileN in allH5:
        (dim1,dim2) = transformString(fileN)
        energies = retrieve_hdf5_data(fileN,'ROOT_ENERGIES')
        bigArrayE[ind] = energies
        bigArray1[ind] = dim1
        bigArray2[ind] = dim2
        ind += 1
#    print(bigArray1,bigArray2)
#    mathematicaListGenerator(bigArray1,bigArray2,bigArrayE)
    gnuSplotCircle(bigArray1,bigArray2,bigArrayE)

def transformString(string):
    [ring,angle] = string.split('Scan')[1].split('.')[0].split('_')
    outputRing = ring.replace('Ring','').replace('P','+').replace('-','.').replace('N','-')
    outputAngle = angle.replace('Angle','').replace('P','+').replace('-','.').replace('N','-')
    ringD = float(outputRing)
    angleD = float(outputAngle)
    dim1 = ringD * np.sin(np.deg2rad(angleD))
    dim2 = ringD * np.cos(np.deg2rad(angleD))
    #print(ringD,angleD,dim1,dim2)
    if abs(dim1) < 0.00000001:
        dim1 = 0
    if abs(dim2) < 0.00000001:
        dim2 = 0
    return(dim1,dim2)

single_inputs = namedtuple("single_input",
                           ("fileXYZ",
                            "vectorX",
                            "vectorY",
                            "linearDisplacement",
                            "circleScan",
                            "graphsGlob",
                            "globExp"))

def main():
    #geom1='CI12.xyz'
    #xf='x'
    #yf='y'
    #circles = 20
    '''
    circles works like:
        generateGeomsAroundConical.py -v CI12.xyz  x  y -c   20    0.1 0.2 0.3 0.4
            the command                   geom    v1 v2   howmany    list of Rs
    '''
    o_inputs = single_inputs("","","",[],[],"","") # defaults
    inp = read_single_arguments(o_inputs)
    #print(inp)
    if inp == o_inputs:
        err("You should use this with some arguments... you know... try -h")
    if inp.graphsGlob == "":
        if inp.globExp == "":
            displaceGeom(inp.fileXYZ,inp.vectorX,inp.vectorY,inp.linearDisplacement,
                     inp.circleScan)
        else:
            scalarProds(inp.globExp,inp.vectorX,inp.vectorY)
    else:
        graphScan(inp.graphsGlob)



if __name__ == "__main__":
    main()

