import numpy as np
from scipy.interpolate import UnivariateSpline

import GeneralFunctions as gf

def addPhase(k,grid,dist,gridN):
    '''
    To add phase to an initial state
    '''
    new = np.empty_like(grid)
    for i in range(gridN):
        new[i] = grid[i] * np.exp(1j*k*dist[i])
    return new

def createInitialState(nstates,gridN,dist,mu,sigma,moment):
    GRIDEMPTY    = np.zeros((nstates,gridN),dtype = complex)
    initialGS    = np.apply_along_axis(gf.gaussian,0,dist,mu,sigma)
    initialGS1   = addPhase(moment,initialGS,dist,gridN)
    norm         = np.linalg.norm(initialGS1)
    normalized   = initialGS1/norm
    GRIDEMPTY[0] = normalized
    return GRIDEMPTY

def reducedMassLiH():
    Li    = 6.941
    H     = 1.00794
    scale = 1836
    eH    = H  * scale
    eLi   = Li * scale
    return (eH*eLi)/(eH+eLi)

def expandDist2(dist,gridNSmall,deltaX,points):
    lower   = dist[0]
    down    = np.empty_like(dist)
    for i in range(gridNSmall):
          down[i] = lower - (i+1)*deltaX
    cutDown = down[0:points]
    revdown = cutDown[::-1]
    return np.concatenate((revdown,dist))

def expandDipo2(dipo,points):
    down = np.zeros_like(dipo)
    return np.concatenate((down[0:points],dipo))

def expandNAC2(NAC,points):
    down = np.zeros_like(NAC)
    return np.concatenate((down[0:points],NAC))

def expandEne2(ene,points):
    down     = (np.zeros_like(ene))
    parabola = (down-3)**2
    return np.concatenate((parabola[0:points],ene))

def extrapolateSx(dist,enePot,points):
    '''
    it takes the potentials and extrapolates new values on the left
    using the UnivariateSpline function from scipy
    The whole +1 -1 thing is a mess, but it does work like this, indeed. Triple checked.
    '''
    points = points + 1
    deltaX = dist[1] - dist[0]
    (gridN,nstates)=enePot.shape
    exp = expandDist2(dist,gridN,deltaX,points)
    final = np.empty((gridN+points-1,nstates))
    for i in range(nstates):
        spl = UnivariateSpline(dist,enePot[:,i])
        spl.set_smoothing_factor(0.0005)
        newExtr = spl(exp)
        cutIt = newExtr[0:points-1]
        newY = np.concatenate((cutIt,enePot[:,i]))
        #gg.makeJustAnother2Dgraph(exp[1:],newY,str(i) + "another.png","lol")
        final[:,i] = newY
    final[0:20,:] = 20.0
    final[-20:,:] = 20.0 # This is also something we need to correct, we want to ADD the well on the right, NOT replacing good data points with it
    return (exp[1:],final)

def extrapolateSxLinear(dist,enePot,points):
    deltaX  = dist[1] - dist[0]
    lower   = dist[0]
    down    = np.empty_like(dist)
    (gridN,nstates) = enePot.shape
    final = np.empty((gridN+points,nstates))
    for i in range(nstates):
        for g in range(gridN):
            down[g] = lower - (g+1)*deltaX
        cutDown = down[0:points]
        revdown = cutDown[::-1]
        newX = np.concatenate((revdown,dist))
        x1 = lower
        y1 = enePot[0,i]
        x2 = revdown[0]
        y2 = 40
        newValues = np.apply_along_axis(lambda t: linebetween2points(t,x1,y1,x2,y2), 0, revdown)
        newY = np.concatenate((newValues,enePot[:,i]))
        final[:,i] = newY
        #gg.makeJustAnother2Dgraph(newX,newY, str(i) + "another.png","lol")
    #final[-20:,:] = 20.0 # This is something we need to correct
    return (newX,final)

def linebetween2points(x,x1,y1,x2,y2):
    return ((x-x1)/(x2-x1))*(y2-y1)+y1

def morse(q, m, u, x ):
    return (q * (np.exp(-2*m*(x-u))-2*np.exp(-m*(x-u))))

def expandDist(dist,gridNSmall,deltaX):
    lower   = dist[0]
    upper   = dist[gridNSmall-1]
    up      = np.empty_like(dist)
    down    = np.empty_like(dist)
    for i in range(gridNSmall):
          up[i]   = upper + (i+1)*deltaX
          down[i] = lower - (i+1)*deltaX
    revdown = down[::-1]
    return np.concatenate((revdown,dist,up))

def expandDipo(dipo):
    uppe = np.zeros_like(dipo)
    down = uppe
    return np.concatenate((down,dipo,uppe))

def expandEne(ene):
    uppe = (np.zeros_like(ene)) + 15
    down = uppe
    return np.concatenate((down,ene,uppe))

def expandEneZero(ene):
    uppe = (np.zeros_like(ene)) + 20
    down = uppe
    central = np.zeros_like(ene)
    return np.concatenate((down,central,uppe))

def expandEneArmonic(ene,k,x0,distSmall, n):
    uppe = (np.zeros_like(ene)) + n
    down = uppe
    central = np.apply_along_axis(armonic,0,distSmall,k,x0)
    other = np.transpose(np.stack((central,central,central,central,central),0)) # HOLY COW
    return (np.concatenate((down,other,uppe)), other)

def armonic(x,k,x0):
    return k*((x-x0)**2)

def absorbingPotential3Right(dist,x0fromRight):
    x0 = dist[-x0fromRight]
    dist3 = (dist-x0)**3
    return dist3.clip(min=0)

if __name__ == "__main__":
     from   astridParser import astridParser2, astridParser2States
     import graph as gg
     (distSmall,eneSmall,dipoSmall,newNAC,newGac) = astridParser2States(4,400)
     (gridNSmall,nstates)           = eneSmall.shape
     deltaX                         = distSmall[1] - distSmall[0]
     points = 40
     newDist = expandDist2(distSmall,gridNSmall,deltaX,points)
     newDip  = expandDipo2(dipoSmall,points)
     newEne  = extrapolateSxLinear(distSmall,eneSmall,points)
     #a = absorbingPotential3Right(newDist,60)
     #gg.makeJustAnother2Dgraph(newDist,a, "absorbing.png" , "lol")
     print("outputfrommain", newEne)


