'''
This scripts generates geometries for norbornadiene given the three famous
angles
'''

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import saveTraj
import numpy as np

def read_single_arguments(single_inputs):
    '''
    This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('-l','--list',
                        nargs='+',
                        help='Vales of phi, gamma and theta',
                        required=True)
    args = parser.parse_args()
    if args.list != None:
        if len(args.list) == 3:
            single_inputs = single_inputs._replace(phi=float(args.list[0]))
            single_inputs = single_inputs._replace(gamma=float(args.list[1]))
            single_inputs = single_inputs._replace(theta=float(args.list[2]))
        else:
            print("WTF is this? Please give me three angles")
    return single_inputs

single_inputs = namedtuple("angles", ("phi","gamma","theta"))

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
    alpha = -0.08447
    beta = -0.38397

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


def main():
    '''
    We need to get the three angles
    and take out a geometry
    '''
    o_inputs = single_inputs(0.0,0.0,0.0)
    inp = read_single_arguments(o_inputs)
    print(inp)
    generateNorbGeometry(inp.phi,inp.gamma,inp.theta)


if __name__ == "__main__":
        main()

