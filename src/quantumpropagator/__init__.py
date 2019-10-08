'''
Init module to import quantumpropagator
'''

from .astridParser import (astridParser)
from .commandlineParser import (readArguments)
from .errors import (err, good, box, warning)
from .GeneralFunctions import (asyncFun, abs2, population, ndprint, singlepop,
                               groundState, gaussian, saveComplex, gaussian2,
                               loadComplex, print2ArrayInColumns, dipoleMoment,
                               fromBohToAng, fromAngToBoh, fromEvtoHart, fromHartoEv, DFT_slow,
                               fromCmMin1toHartree, calculateGradientOnMatrix0,
                               chunksOf, chunksOfList, calcBond, calcAngle,
                               calcDihedral, massOf, saveTraj, scanvalues, fromHartreetoCmMin1,
                               printMatrix2D, readGeometry, labTranform, labTranformA,
                               stringTransformation3d, loadInputYAML, fromAuToFs,
                               createTabellineFromArray,fromFsToAu, printDict, readDirectionFile,
                               bring_input_to_AU, printDictKeys, printProgressBar, fromCmMin1toFs,
                               fromLabelsToFloats,labTranformReverse,labTranformReverseA,
                               fromFloatsToLabels,equilibriumIndex,generateNorbGeometry,
                               pickleLoad, pickleSave, find_numpy_index_minumum,
                               create_list_of_colors)
from .GeneralFunctionsSystem import (cd, ensure_dir_Secure, ensure_dir, create_enumerated_folder)
from .graph import (getLabels, LiHAstLab, LiHLab, createStatesLab,
                    makeJustAnother2Dgraph, grapPulse, makeJustAnother2DgraphComplex,
                    makeJustAnother2DgraphComplexALLS, makeJustAnother2DgraphComplexSINGLE,
                    dipoleMatrixGraph1d, makeMultiPlotMatrix, make2DgraphTranspose,
                    makePulseSinglePointGraph, createHeatMapfromH5, createSingleHeatmap,
                    createCoherenceHeatMapfromH5, ccmu, createHistogram,
                    makeMultiLineDipoleGraph, makeJustAnother2DgraphMULTI,
                    mathematicaListGenerator, gnuSplotCircle, heatMap2dWavefunction,graphic_Pulse)
from .GridIntegrator import (grid1DIntegrationAstrid)
from .h5Reader import (retrieve_hdf5_keys, retrieve_hdf5_data, readAllhdf51D,
                       writeH5file, correctSignThem, correctSignFromRelativeToAbsolute,
                       findCorrectionNumber, correctSign, secondCorrection,
                       npArrayOfFiles, writeH5fileDict, readWholeH5toDict)
from .initialConditions import (addPhase, createInitialState, reducedMassLiH,
                                expandDist2, expandDipo2, expandNAC2, expandEne2, extrapolateSx,
                                extrapolateSxLinear, linebetween2points, morse, expandDist,
                                expandDipo, expandEne, expandEneZero, expandEneArmonic, armonic,
                                absorbingPotential3Right)
from .molcas import (launchSPfromGeom, writeRassiLiHInput, LiHxyz, WaterXyz,
                     generateLiHxyz, generateWater)
from .Propagator import (rk4Ene1dSLOW, rk4Ene3d, derivative3d, derivative1d, HamiltonianEle1d,
                         NuclearKinetic1d, doubleDeronK, singleDerOnK,
                         createXaxisReciprocalspace1d, divideByTwo, calculateTotal,
                         rk6Ene1dSLOW, EULER1d,derivative1dPhi, derivative1dGam,
                         derivative2dGamThe, derivative2dGamTheMu, derivative3dMu)
from .EMPulse import (specificPulse, userPulse, varPulseZ, specificPulseZero, component,
                      envel, pulse)
from .SinglePointPropagator import (single_point_propagation, printEvenergy)
from .Densities import (give_me_swapd_oxiallyl, transform_numpy_into_format)
from .KineticMadness import (calc_g_G, calc_s_mat)
from .TDPropagator import propagate3D, restart_propagation, calculate_stuffs_on_WF
from .IpythonJupyter import *
from .CPropagator import (Crk4Ene3d, CextractEnergy3dMu, Cderivative3dMu, Cenergy_1D_Phi,
                          Cderivative_1D_Phi,Cenergy_1D_Gam, Cderivative_1D_Gam,Cenergy_1D_The,
                          Cderivative_1D_The, pulZe,version_Cpropagator, CextractMomentum3d,
                          calculate_dipole_fast_wrapper, fft_artisanal, Cderivative3dMu_reverse_time)

