'''
This scripts generates geometries for norbornadiene given the three famous
angles
'''

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import (saveTraj, readGeometry)
import numpy as np

def read_single_arguments(single_inputs):
    '''
    This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('-l','--list',
                        nargs='+',
                        help='Vales of phi, gamma and theta',
                        )
    parser.add_argument('-i','--info',
                        dest="i",
                        type=str,
                        help='gives info on a certain geometry',
                        )
    args = parser.parse_args()
    if args.i != None:
        single_inputs = single_inputs._replace(geomfile=args.i)
    if args.list != None:
        if len(args.list) == 9:
            single_inputs = single_inputs._replace(phi0=float(args.list[0]))
            single_inputs = single_inputs._replace(phiF=float(args.list[1]))
            single_inputs = single_inputs._replace(phiD=float(args.list[2]))
            single_inputs = single_inputs._replace(gamma0=float(args.list[3]))
            single_inputs = single_inputs._replace(gammaF=float(args.list[4]))
            single_inputs = single_inputs._replace(gammaD=float(args.list[5]))
            single_inputs = single_inputs._replace(theta0=float(args.list[6]))
            single_inputs = single_inputs._replace(thetaF=float(args.list[7]))
            single_inputs = single_inputs._replace(thetaD=float(args.list[8]))
        elif len(args.list) == 3:
            single_inputs = single_inputs._replace(phi0=float(args.list[0]))
            single_inputs = single_inputs._replace(gamma0=float(args.list[1]))
            single_inputs = single_inputs._replace(theta0=float(args.list[2]))
        else:
            print("WTF is this? Please give me the nine things, now...")
    return single_inputs

single_inputs = namedtuple("angles",
                          ("phi0","gamma0","theta0",
                           "phiF","gammaF","thetaF",
                           "phiD","gammaD","thetaD","geomfile"))

def generateNorbGeometry(phi,gam,the):
    '''
    This function generates an xyz given the value of the three angles
    phi,gam,the :: Double  <- the three angles
    '''
    # Norbornadiene_P001-000_N001-000_P001-000
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
    bridgeDis = 2.337775 # distance
    the2 = np.deg2rad(the/2)
    phi2 = np.deg2rad(phi)
    gam2 = np.deg2rad(gam)

    # This is the code for the ACTUAL angle between the three carbons. I need to modify the
    # code.

    xC1 = -rBond * np.sin(phi2+the2)
    yC1 = L + rBond * np.cos(phi2+the2) * -np.sin(gam2)
    zC1 = -rBond * np.cos(phi2+the2) * np.cos(gam2)

    xC2 = -rBond * np.sin(phi2-the2)
    yC2 = L + rBond * np.cos(phi2-the2) * -np.sin(gam2)
    zC2 = -rBond * np.cos(phi2-the2) * np.cos(gam2)

    xC3 = rBond * np.sin(phi2+the2)
    yC3 = -L + rBond * np.cos(phi2+the2) * np.sin(gam2)
    zC3 = -rBond * np.cos(phi2+the2) * np.cos(gam2)

    xC4 = rBond * np.sin(phi2-the2)
    yC4 = -L + rBond * np.cos(phi2-the2) * np.sin(gam2)
    zC4 = -rBond * np.cos(phi2-the2) * np.cos(gam2)

# this attempt is done with mathematica, keeps the angle at the projection
# DOes well describe gamma and theta (problems with phi)
# does NOT conserve the distance CC when changing phi

#    a = phi2+the2
#    b = gam2
#
#    de1 = np.cos(a)**2
#    de2 = np.cos(b)**2
#    de3 = de1 * de2
#    deno = np.sqrt(de1 + de2 - de3)
#
#
#    xC1 = - (rBond * np.cos(b) * np.sin(a))/deno
#    yC1 = L - (rBond * np.cos(a) * np.sin(b))/deno
#    zC1 = - (rBond * np.cos(a) * np.cos(b))/deno
#
#    xC3 = (rBond * np.cos(b) * np.sin(a))/deno
#    yC3 = - L + (rBond * np.cos(a) * np.sin(b))/deno
#    zC3 = - (rBond * np.cos(a) * np.cos(b))/deno
#
#    a = phi2-the2
#    b = gam2
#
#    de1 = np.cos(a)**2
#    de2 = np.cos(b)**2
#    de3 = de1 * de2
#    deno = np.sqrt(de1 + de2 - de3)
#
#    xC2 = - (rBond * np.cos(b) * np.sin(a))/deno
#    yC2 = L - (rBond * np.cos(a) * np.sin(b))/deno
#    zC2 = - (rBond * np.cos(a) * np.cos(b))/deno
#
#    xC4 = (rBond * np.cos(b) * np.sin(a))/deno
#    yC4 = - L + (rBond * np.cos(a) * np.sin(b))/deno
#    zC4 = - (rBond * np.cos(a) * np.cos(b))/deno


    # this is angle CH on the zx plane, calculated from open optimized geometry
    # using the coordinate of a C and a H -> np.arcsin((x2-x1)/(np.sqrt((x2-x1)**2+(z2-z1)**2)))
    alpha = -0.913346
    # same as the other, but projected on plane zy and thus with formula
    # np.arcsin((y2-y1)/(np.sqrt((y2-y1)**2+(z2-z1)**2)))
    beta = -0.899358

    xH1 = xC1 + (chBond * np.sin(alpha))
    yH1 = yC1 - (chBond * np.cos(alpha) * np.sin(beta))
    zH1 = zC1 - (chBond * np.cos(alpha) * np.cos(beta))

    xH2 = xC2 - (chBond * np.sin(alpha))
    yH2 = yC2 - (chBond * np.cos(alpha) * np.sin(beta))
    zH2 = zC2 - (chBond * np.cos(alpha) * np.cos(beta))

    xH3 = xC3 - (chBond * np.sin(alpha))
    yH3 = yC3 + (chBond * np.cos(alpha) * np.sin(beta))
    zH3 = zC3 - (chBond * np.cos(alpha) * np.cos(beta))

    xH4 = xC4 + (chBond * np.sin(alpha))
    yH4 = yC4 + (chBond * np.cos(alpha) * np.sin(beta))
    zH4 = zC4 - (chBond * np.cos(alpha) * np.cos(beta))

    newAtoms = [[xC1,yC1,zC1], [xC2,yC2,zC2], [xC3,yC3,zC3], [xC4,yC4,zC4],
                [xH1,yH1,zH1], [xH2,yH2,zH2], [xH3,yH3,zH3], [xH4,yH4,zH4]]
    new = np.append(fixed,newAtoms,0)
    atomTN = atomT + ['C', 'C', 'C', 'C', 'H', 'H', 'H', 'H']
    print(new)
    # saveTraj works on LIST of geometries, that is why the double list brackets
    saveTraj(np.array([new]),atomTN,fn)

def getAnglesFRomGeometry(fn):
    '''
    given a geometry it gives back the angles theta and gamma
    (to be improved)
    fn :: String <- filePath
    '''
    (natom,title,atomType,geom) = readGeometry(fn)
    # in my problem this is the Theta angle
    atom1 = 2
    atom2 = 11
    [[x1,y1,z1],[x2,y2,z2]] = [geom[atom1-1],geom[atom2-1]]
    theta = np.rad2deg(np.arcsin((x2-x1)/(np.sqrt((x2-x1)**2+(z2-z1)**2))))
    gamma = np.rad2deg(np.arcsin((y2-y1)/(np.sqrt((y2-y1)**2+(z2-z1)**2))))
    string = '''
    Geomtry file:   {}

    Theta = {:7.3f}  <- this counts phi, also
    Gamma = {:7.3f}
    '''.format(fn,theta,gamma)
    print(string)

def main():
    '''
    We need to get the three angles
    and take out a geometry
    if you give the script 3 numbers, it will create a geometry, otherwise 9
    numbers will create a 3x3 grid
    '''
    o_inputs = single_inputs(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"")
    inp = read_single_arguments(o_inputs)
    print('\n')
    print(inp)
    print('\n\n')
    if inp.geomfile != "":
        getAnglesFRomGeometry(inp.geomfile)
    elif (inp.phiD == 0.0) and (inp.phiD == 0.0) and (inp.phiD == 0.0):
        generateNorbGeometry(inp.phi0,inp.gamma0,inp.theta0)
    else:
        for phi in np.linspace(inp.phi0,inp.phiF,inp.phiD):
            for gamma in np.linspace(inp.gamma0,inp.gammaF,inp.gammaD):
                for theta in np.linspace(inp.theta0,inp.thetaF,inp.thetaD):
                    generateNorbGeometry(phi,gamma,theta)

if __name__ == "__main__":
        main()

