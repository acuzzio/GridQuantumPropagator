'''
This Script will generate geometries around the conical intersection, linear
and circle.

It needs a geometry file (xyz) and two vectors (that can be read by np.loadtxt)
'''

import numpy as np
from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import (readGeometry,saveTraj,err)

def read_single_arguments(single_inputs):
    '''
     This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-v", "--vectors",
                        dest="v",
                        nargs='+',
                        help="the 3 files: geometry and branching plane vectors")
    parser.add_argument("-l", "--linear",
                        dest="l",
                        nargs='+',
                        help="parameter for the linear displacement.")
    args = parser.parse_args()
    if args.v != None:
        if len(args.v) == 3:
            [geom,grad,der] = args.v
            single_inputs = single_inputs._replace(fileXYZ=geom)
            single_inputs = single_inputs._replace(vectorX=grad)
            single_inputs = single_inputs._replace(vectorY=der)
        else:
            err('this takes 3 arguments: the goemetry and the two vectors')
    if args.l != None:
        single_inputs = single_inputs._replace(linearDisplacement=args.l)
    return single_inputs

def displaceGeom(geom1,xf,yf,linearArgs,circleArgs):
    GD = np.loadtxt(xf)
    NA = np.loadtxt(yf)
    (natoms,title,atomTN,geom) = readGeometry(geom1)
    #print(natoms,title,atomType,geom,GD,NA)
    labelRoot = geom1.split[0]
    if linearArgs != []:
        distance = 0.1
        npoints = 11
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
        circles = 20
        Rlist = [0.02]
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


single_inputs = namedtuple("single_input",
                           ("fileXYZ",
                            "vectorX",
                            "vectorY",
                            "linearDisplacement",
                            "circleScan",
                            "graphs"))

def main():
    #geom1='CI12.xyz'
    #xf='x'
    #yf='y'
    #circles = 20
    o_inputs = single_inputs("","","",[],[],False)
    inp = read_single_arguments(o_inputs)
    if inp.graphs:
        print(inp)
    else:
        print(inp)
        print('Graphs')




if __name__ == "__main__":
    main()

