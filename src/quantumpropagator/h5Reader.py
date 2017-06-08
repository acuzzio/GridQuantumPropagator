
import numpy as np
import h5py
import multiprocessing as mp
import glob

processNumber=8

def retrieve_hdf5_keys(path_hdf5):
    '''
    Get key values from file path_hdf5.
    '''
    try:
       with h5py.File(path_hdf5, 'r') as hf:
          return list(hf.keys())
    except FileNotFoundError:
       msg = "there is not HDF5 file with that name"
       raise RuntimeError(msg)

def retrieve_hdf5_data(path_hdf5, paths_to_prop):
    '''
    Read Numerical properties from ``paths_hdf5``.
    :params path_hdf5: Path to the hdf5 file
    :type path_hdf5: string
    :returns: numerical array
    '''
    try:
        with h5py.File(path_hdf5, 'r') as f5:
            if isinstance(paths_to_prop, list):
                return [f5[path].value for path in paths_to_prop]
            else:
                return f5[paths_to_prop].value
    except KeyError:
        msg = "There is not {} stored in the HDF5\n".format(paths_to_prop)
        raise KeyError(msg)
    except FileNotFoundError:
        msg = "there is not HDF5 file containing the numerical results"
        raise RuntimeError(msg)

def readAllhdf51D(globalExp,nstates,proparraylength):
    '''
    It reads a grid of files in a certain global expression by globalExp
    '''
    list_of_files = npArrayOfFiles(globalExp) # this will be subst by the shaping grid function
    shapeFiles    = list_of_files.shape
    dp            = np.empty(shapeFiles + (proparraylength, nstates, nstates)) # MOLCAS DEPENDENT watch out this proparraylength BOUND
    en            = np.empty(shapeFiles + (nstates,))
    ge            = np.empty(shapeFiles + (2,3)) # THIS IS LIH BOUND !!
    with mp.Pool(processes=processNumber) as p:
        promises = [p.apply_async(retrieve_hdf5_data, args=(list_of_files[fileS], ['MLTPL','SFS_ENERGIES','CENTER_COORDINATES']))
                    for fileS in np.ndindex(shapeFiles)]
        for (p, i) in zip(promises, np.ndindex(shapeFiles)):
            (dp[i],en[i],ge[i]) = p.get()
    realDP = dp[:,0:3,::] # MOLCAS BOUND
    return (realDP, en, ge)

def writeH5file(fn, tuplesLabelValues):
    '''
    writes a h5 file with name fn and [(label, value)] structure
    '''
    with h5py.File(fn, 'w') as hf:
            for (label,value) in tuplesLabelValues:
                hf.create_dataset(label, data=value)


def correctSignThem(folder, nstates, proparraylength):
    '''
    corr       -> array of 1 and -1 for the sign correction. dimension :: (nfiles,nstates)
    tdMatrices -> the dipole transition matrices.            dimension :: (nfiles,proparraylength,nstates,nstates)

    '''
    abscorr                            = findCorrectionNumber(folder,nstates)
    (gridN,_)                          = abscorr.shape
    abscorrT                           = np.transpose(abscorr)
    corrT                              = np.empty(abscorrT.shape)
    for i in range(nstates):
        corrT[i] = correctSignFromRelativeToAbsolute(abscorrT[i])
    corr = np.transpose(corrT)
    (tdMatrices, energies, geoms) = readAllhdf51D(folder + "*.rassi.h5", nstates,proparraylength)
    newdipoleM                    = np.empty((gridN, 3, nstates, nstates)) 
    for fileN in range(gridN):
        tdMatrix   = tdMatrices[fileN]
        dipole     = tdMatrix[0:3]   # Molcas BOUND I need only the first three
        correction = corr[fileN]
        for k in range(3):
            for i in range(nstates):
                for j in range(nstates):
                    correctionforElement  = correction[i]*correction[j]
                    newdipoleM[fileN,k,i,j] = dipole[k,i,j]*correctionforElement
        tupleZ   = [('MLTPL', newdipoleM[fileN]), ('SFS_ENERGIES', energies[fileN]), ('CENTER_COORDINATES', geoms[fileN])]
        filename = folder + 'fileOut' + '{:03}'.format(fileN) + '.h5'
        writeH5file(filename, tupleZ) 



def correctSignFromRelativeToAbsolute(vector):
    '''
    This function assures that we correct the Nth vector sign based on the CORRECTED PRECEDENT sign
    so, if there is a [+1,-1,+1,-1,-1] situation, I want this vector to be [1,-1,-1,+1,-1].
    '''
    boolean = True # true means that we are in the + domain...
    gridN   = vector.size
    new     = np.empty(gridN)
    for i in range(gridN):
        if vector[i] == 1.:
            if boolean == True:
                new[i]=1.
            else:
                new[i]=-1.
        else:
            if boolean == True:
                new[i]=-1.
                boolean = not boolean
            else:
                new[i]=1.
                boolean = not boolean
    return(new)


def findCorrectionNumber(folder,nstates):
    rasscfFiles = npArrayOfFiles(folder + '*.rasscf.h5')
    nfiles      = rasscfFiles.size 
    result      = np.empty((nfiles,nstates))
    result[0,:] = 1.0
    for i in range(nfiles)[1:]:
        result[i] = correctSign(rasscfFiles[i-1],rasscfFiles[i])
    print(result)
    return result


def correctSign(rasscfh51, rasscfh52):
    '''
    given 2 rasscf.h5 files, this will return a vector of {1;-1} elements depending on the scalar
    product between the two states.
    '''
    CI1  = retrieve_hdf5_data(rasscfh51, 'CI_VECTORS')
    CI2  = retrieve_hdf5_data(rasscfh52, 'CI_VECTORS')
    dots = np.sum(CI1*CI2, axis=1)
    (dim,)  = dots.shape
    new  = np.empty(dim)
    for i in range(dim):
        if dots[i] < 0:
           new[i] = -1.0
        else:
           new[i] = 1.0
    return new

def secondCorrection(globalE, outlabel, nstates, proparraylength):
    '''
    This is an old correction with absolute value
    '''
    files                         = npArrayOfFiles(globalE)
    (tdMatrices, energies, geoms) = readAllhdf51D(globalE, nstates, proparraylength)
    (a,_)                         = np.unravel_index(energies.argmin(), energies.shape)
    reference                     = np.sign(tdMatrices[a])
    (filenumber,_,_)              = geoms.shape
    newdipoleM                    = np.empty((filenumber,3,nstates,nstates))   
    for fileN in range(filenumber):
        tdMatrix   = tdMatrices[fileN]
        for k in range(3):
            for i in range(nstates):
                for j in range(nstates):
                    correctionforElement  = reference[k,i,j]
                    newdipoleM[fileN,k,i,j] = abs(tdMatrix[k,i,j])*correctionforElement
        tupleZ   = [('MLTPL', newdipoleM[fileN]),('SFS_ENERGIES', energies[fileN]), ('CENTER_COORDINATES', geoms[fileN])]
        filename = folder + outlabel + '{:03}'.format(fileN) + '.h5'
        writeH5file(filename, tupleZ)


def npArrayOfFiles(globalE):
    '''
    desugaring
    '''
    return np.array(sorted(glob.glob(globalE)))

if __name__ == "__main__":
        folder = 'Scanh5Files/'
        correctSignThem(folder,9,6)
