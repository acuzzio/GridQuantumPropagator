'''
This scripts generates geometries for norbornadiene given the three famous
angles
'''

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import (saveTraj, readGeometry, calcAngle, err, good,
                               ndprint, generateNorbGeometry)
import numpy as np

def read_single_arguments(single_inputs):
    '''
    This funcion reads the command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument('-l','--list',
                        nargs='+',
                        help='Values of phi, gamma and theta',
                        )
    parser.add_argument('-n','--newlist',
                        nargs='+',
                        help='Values of phi, gamma and theta, where phi is linear',
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
            err("WTF? This takes 3 numbers (single point) or 9 (for a line/box)")
    if args.newlist != None:
        single_inputs = single_inputs._replace(linearphi=True)
        if len(args.newlist) == 9:
            single_inputs = single_inputs._replace(phi0=float(args.newlist[0]))
            single_inputs = single_inputs._replace(phiF=float(args.newlist[1]))
            single_inputs = single_inputs._replace(phiD=float(args.newlist[2]))
            single_inputs = single_inputs._replace(gamma0=float(args.newlist[3]))
            single_inputs = single_inputs._replace(gammaF=float(args.newlist[4]))
            single_inputs = single_inputs._replace(gammaD=float(args.newlist[5]))
            single_inputs = single_inputs._replace(theta0=float(args.newlist[6]))
            single_inputs = single_inputs._replace(thetaF=float(args.newlist[7]))
            single_inputs = single_inputs._replace(thetaD=float(args.newlist[8]))
        elif len(args.newlist) == 3:
            single_inputs = single_inputs._replace(phi0=float(args.newlist[0]))
            single_inputs = single_inputs._replace(gamma0=float(args.newlist[1]))
            single_inputs = single_inputs._replace(theta0=float(args.newlist[2]))
        else:
            err("WTF? This takes 3 numbers (single point) or 9 (for a line/box)")
    return single_inputs

single_inputs = namedtuple("angles",
                          ("phi0","gamma0","theta0",
                           "phiF","gammaF","thetaF",
                           "phiD","gammaD","thetaD","geomfile","linearphi"))

def getAnglesFRomGeometry(fn):
    '''
    given a geometry it gives back the angles phi, theta and gamma
    (to be improved)
    fn :: String <- filePath
    '''
    (natom,title,atomType,geom) = readGeometry(fn)
    # in my problem this is the Theta angle
    atom1 = 2
    atom2 = 11
    atom3 = 10
    atom4 = 9
    atom5 = 8
    [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3],[x4,y4,z4],[x5,y5,z5]] = [
            geom[atom1-1],geom[atom2-1],geom[atom3-1],geom[atom4-1],geom[atom5-1]]
    theta10 = np.rad2deg(np.arcsin((x2-x1)/(np.sqrt((x2-x1)**2+(z2-z1)**2))))
    theta11 = np.rad2deg(np.arcsin((x3-x1)/(np.sqrt((x3-x1)**2+(z3-z1)**2))))
    theta9 = np.rad2deg(np.arcsin((x4-x1)/(np.sqrt((x4-x1)**2+(z4-z1)**2))))
    theta8 = np.rad2deg(np.arcsin((x5-x1)/(np.sqrt((x5-x1)**2+(z5-z1)**2))))
    #theta = theta12 - theta11
    gamma8 = 90 - calcAngle(geom,8,3,2)
    gamma9 = 90 - calcAngle(geom,9,3,2)
    gamma10 = 90 - calcAngle(geom,10,2,3)
    gamma11 = 90 - calcAngle(geom,11,2,3)
    #gammaP = np.rad2deg(np.arcsin((y2-y1)/(np.sqrt((y2-y1)**2+(z2-z1)**2))))
    gavg = (gamma11 + gamma10 + gamma9 + gamma8)/4
    tavg = (abs(theta11) + abs(theta10) + abs(theta9) + abs(theta8))/2
    string = '''
    Geomtry file:   {}

    Gamma8 = {:7.3f}
    Gamma9 = {:7.3f}
    Gamma10 = {:7.3f}
    Gamma11 = {:7.3f}
    GammaAVG = {:7.3f}

    Theta8 = {:7.3f}
    Theta9 = {:7.3f}
    Theta10 = {:7.3f}
    Theta11 = {:7.3f}
    ThetaAVG = {:7.3f}
    '''.format(fn,gamma8,gamma9,gamma10,gamma11,gavg,theta8,theta9,theta10,theta11,tavg)
    print(string)

def main():
    '''
    We need to get the three angles
    and take out a geometry
    if you give the script 3 numbers, it will create a geometry, otherwise 9
    numbers will create a 3x3 grid
    '''
    o_inputs = single_inputs(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,"",False)
    inp = read_single_arguments(o_inputs)
    print('\n')
    print(inp)
    print('\n\n')
    if inp.geomfile != "":
        getAnglesFRomGeometry(inp.geomfile)
    elif (inp.phiD == 0.0) and (inp.phiD == 0.0) and (inp.phiD == 0.0):
        generateNorbGeometry(inp.phi0,inp.gamma0,inp.theta0)
    else:
        phiRange = np.linspace(inp.phi0,inp.phiF,inp.phiD)
        gammaRange = np.linspace(inp.gamma0,inp.gammaF,inp.gammaD)
        thetaRange = np.linspace(inp.theta0,inp.thetaF,inp.thetaD)
        for phi in phiRange:
            for gamma in gammaRange:
                for theta in thetaRange:
                    generateNorbGeometry(phi,gamma,theta)
        outStr1 = ndprint(phiRange,format_string='{:+08.3f}')
        outStr1 = outStr1.replace('-','N').replace('.','-').replace('+','P')
        outStr2 = ndprint(gammaRange,format_string='{:+08.3f}')
        outStr2 = outStr2.replace('-','N').replace('.','-').replace('+','P')
        outStr3 = ndprint(thetaRange,format_string='{:+08.3f}')
        outStr3 = outStr3.replace('-','N').replace('.','-').replace('+','P')
        outForBash = '''
        Phi:
{}

        Gamma:
{}

        Theta:
{}

        '''.format(outStr1,outStr2,outStr3)
        print(outForBash)
    good('REMEMBER THAT THE HYDROGENS ARE PUT WITH CARTESIAN')

if __name__ == "__main__":
        main()
