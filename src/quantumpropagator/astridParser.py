'''
This module is a parser for a specific dataset from literature. It serves the purpose
to test the propagator on a grid.
'''

from os.path import join
import numpy as np

from quantumpropagator.GeneralFunctions import calculateGradientOnMatrix0

def astridParser(cutAt,gridN,folder):
    '''
    This is the function that loads the initial files. It returns several arrays from files:
    dist     :: arr(gridN)
    newEne   :: arr(nstates,gridN)
    newDipo  :: arr(gridN,xyz,nstates,nstates)  <- this needs to become a vector
    newNAC   :: arr(gridN,nstates,nstates)      <- this as well (gridN,xyz,nstates,nstates)
    newGac   :: arr(gridN,nstates,nstates)      <- this as well (gridN,xyz,nstates,nstates)
    Along the grid long gridN, dist is the x coordinate, newEne are the potential energies
    newDipo the dipole moments, newNAC the adiabatic couplings, newGAC the derivative of newNAC
    '''
    eneF         = join(folder, 'potentials_energies.txt')
    dipF         = join(folder, 'potentials_transition_dipoles_z.txt')
    perDipF      = join(folder, 'potentials_perm_dipoles_z.txt')
    neighNAC     = join(folder, 'potentials_neihboring_nacs.txt')
    nstates      = 5
    nstateI      = nstates + 1
    nstatesRange = np.arange(nstates-1)
    DistEne      = np.loadtxt(eneF)
    dist         = DistEne[:gridN,0]
    ene          = DistEne[:gridN,1:nstateI]
    dipAllCol    = np.loadtxt(dipF)
    dipupperDiag = dipAllCol[:,1:11]  # this is the tricky one
    perDipA      = np.loadtxt(perDipF)
    dopDiag      = perDipA[:,1:nstateI]
    dipo         = np.empty((gridN,1,nstates,nstates))
    NACA         = np.loadtxt(neighNAC)
    NACACUT      = NACA[:gridN,1:nstateI]
    NAC          = np.empty((gridN,nstates,nstates))
    for i in range(gridN):
        upper      = dipupperDiag[i]
        diag       = dopDiag[i]
        new        = np.zeros((nstates, nstates))
        new[np.triu_indices(nstates, 1)] = upper
        new += new.T
        new[np.diag_indices(nstates)] = diag
        dipo[i,0] = new
        # From the problem of Astrid, this matrix is an offdiagonal one,
        # zero except when [nstatesRange,nstatesRange+1] and [nstatesRange+1,nstatesRange]
        offdiagonalNAC = NACACUT[i]
        NAC[i,nstatesRange,nstatesRange+1]=offdiagonalNAC
        NAC[i,nstatesRange+1,nstatesRange]=-offdiagonalNAC
    newEne  = ene[:,0:cutAt]
    newDipo = dipo[:,:,0:cutAt,0:cutAt]
    newNAC  = NAC[:,0:cutAt,0:cutAt]
    newGac  = calculateGradientOnMatrix0(newNAC,dist)
    return(dist, newEne, newDipo, newNAC, newGac)


# quick reminder of how triu works.
#dm = np.arange((15))
#tri = np.zeros((5, 5))
#tri[np.triu_indices(5, 0)] = dm  <- with diagonal
#tri[np.triu_indices(5, 1)] = dm  <- without diagonal


if __name__ == "__main__":
    (dist,newEne,newDipo,newNAC,newGac) = astridParser(4,400,'/home/alessio/y-LiegiInitialReport/ast2')
    print(newGac)
    print(dist.shape, newEne.shape, newDipo.shape, newNAC.shape, newGac.shape)

