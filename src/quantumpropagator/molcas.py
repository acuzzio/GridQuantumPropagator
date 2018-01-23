
import os
import numpy as np
from shutil import copy
from subprocess import call

import quantumpropagator.GeneralFunctionsSystem as gfs


def launchSPfromGeom(geom):
    '''
    geom -> xyz name (with path)
    given a xyz into a folder it launches the corresponding molcas calculation
    '''
    folder_path   = ('.').join(geom.split('.')[:-1])
    gfs.ensure_dir(folder_path)
    copy(geom,folder_path)
    projectN    = ('.').join(os.path.basename(geom).split('.')[:-1])
    inputname   = folder_path + "/" + projectN + ".input"
    writeRassiLiHInput(inputname)
    with gfs.cd(folder_path):
         call(["/home/alessio/bin/LaunchMolcas", inputname])


def writeRassiLiHInput(inputname):
    content = """ >>> LINK FORCE $Project.JobOld JOBOLD

&Gateway
  coord=$Project.xyz
  basis=6-31PPGDD
  group=NoSym

&Seward

&Rasscf
  nactel = 4 0 0
  ras2 = 20
  inactive = 0
  ciroot = 9 9 1
  prwf = 0.0

&Rassi
  mees

&grid_it
  all

>> COPY $Project.rassi.h5 $HomeDir
>> COPY $Project.JobIph $HomeDir
"""
    with open(inputname, 'w') as f:
         f.write(content)

def LiHxyz(folder,distance,label):
    lihxyz = """    2

  Li     0.00000000     0.00000000     0.00000000
   H     {distance:5.8f}     0.00000000     0.00000000
"""
    context = {"distance":distance}
    fnN = folder + 'LiH' + label + '.xyz'
    with open(fnN, 'w') as f:
        f.write(lihxyz.format(**context))

def WaterXyz(folder,distance,label):
    watxyz = """   3

 O     0.000000     0.000000     0.000000
 H     {distance:7.6f}     0.000000     0.000000
 H    -0.251204    -0.905813     0.000000
    """
    return watxyz

def generateLiHxyz(outfolder, rangearg):
    doubleList = list(np.arange(*rangearg).tolist())
    for tup in enumerate(doubleList):
        (label,dist) = tup
        label3 = '%03i' % label
        LiHxyz(outfolder, dist, label3)

def generateWater(outfolder, rangearg):
    doubleList = list(np.arange(*rangearg).tolist())
    for tup in enumerate(doubleList):
        (label,dist) = tup
        label3 = '%03i' % label
        WaterXyz(outfolder, dist, label3)

if __name__ == "__main__":
    print('lol!!!')
#    import quantumpropagator.h5Reader as hf
#    import quantumpropagator.GeneralFunctions as gf
#    fn = 'Grid_119.648_000.000.rassi.h5'
#    overlapsM = hf.retrieve_hdf5_data(fn, 'ORIGINAL_OVERLAPS')
#    (dim, _ ) = overlapsM.shape
#    nstates = dim // 2
#    overlaps = overlapsM[nstates:,:nstates]
#    gf.printMatrix2D(overlaps,2)
#    arrayOneD = compressColumnOverlap(overlaps)
#    correctionMatrix = gf.createTabellineFromArray(arrayOneD)
#    print(arrayOneD)
#    print(correctionMatrix)

    #generateLiHxyz('XyzS/', (0.7,4.0,0.1))
    #fns = sorted(glob.glob('XyzS/*'))
    #for fileN in fns:
    #    launchSPfromGeom(fileN)

