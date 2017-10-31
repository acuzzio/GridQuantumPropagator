'''
This is the module where the program generates graphics
'''

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import quantumpropagator.h5Reader as h5
import quantumpropagator.GeneralFunctions as gf
import quantumpropagator.EMPulse as pp


def getLabels(string):
    '''
    A dictionary for different labels/different systems.
    It will be upgraded to something read from an input file
    '''
    supportedSystems = {"LiH": LiHLab,
                        "LiHAst": LiHAstLab,
                        "CisButa": CisButa}
    return (supportedSystems[string])


def CisButa():
    ''' labels of cisbutadiene '''
    return (createStatesLab(['a','a','b','a','a','b']),
            ['b','g','r','c','m','y']) # THIS IS NOT THE RIGHT ORDER!!

def LiHAstLab():
    ''' The labels of the LiH of the test astrid dataset '''
    return (createStatesLab(['s','s','s','s','s']), ['b','g','r','c','m'])

def LiHLab():
    ''' The labels for the Lih - sigma pi for the 0D problem '''
    colorsA = ['b','g','r','c','m','y','k','#808080', '#606060']
    statesSym = ['s','s','p','p','s','s','p','p','s']
    statesLab = createStatesLab(statesSym)
    return (statesLab,colorsA)


def createStatesLab(statesSym):
    '''
    This function takes the label string and creates the correct LAteX label to use in the graph key
    '''
    labels = {'s' : '\Sigma', 'p' : '\Pi', 'a' : 'A', 'b' : 'B'}
    init   = {'s' : 0, 'p' : 1, 'a' : 1, 'b' : 1}
    counter_d = {}
    correctlabel = []
    for lab in statesSym:
        if not lab in counter_d:
           counter_d[lab] = init[lab]
        else:
           counter_d[lab] += 1
        correctlabel.append(r'$' + labels[lab] + '_' + str(counter_d[lab]) + '$')
    return(correctlabel)


def makeJustAnother2Dgraph(xs,ys,fn,labl):
    '''
    xs :: np.array[Double]
    yx :: np.array[Double]
    fn :: FilePath
    labl :: String - the name on the key
    '''
    transp    = False
    my_dpi    = 150
    ratio     = (16, 9)
    fig, ax1  = plt.subplots(figsize=ratio)
    plt.plot(xs, ys,linewidth=3.0,label=labl)
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')


def grapPulse(totaltime, dt, Ed, omega, sigma, phi, t0, fn):
    '''
    This makes a graph of a complete pulse from a single direction
    '''
    timesArray = np.arange(totaltime) * dt
    gaus = np.apply_along_axis(pp.envel,0,timesArray,Ed,sigma,t0)
    puls = np.apply_along_axis(pp.pulse,0,timesArray,Ed,omega,sigma,phi,t0)
    transp = False
    my_dpi = 150
    ratio = (16, 9)
    fig, ax1 = plt.subplots(figsize=ratio)
    ax1.set_ylabel('E A.U. (Hartree)')
    ax1.set_xlabel('Time A.U.')
    plt.plot(timesArray, gaus, linewidth=2.0, label="Envelope")
    plt.plot(timesArray, puls, linewidth=2.0, label="Pulse")
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')


def makeJustAnother2DgraphComplex(xs,ys,fn,labl):
    transp = False
    my_dpi = 150
    ratio = (16, 9)
    imgL = labl + " Imag"
    reaL = labl + " Real"
    fig, ax1 = plt.subplots(figsize=ratio)
    ax1.set_ylim([-1,1])
    plt.plot(xs, np.real(ys), linewidth=3.0, label=reaL)
    plt.plot(xs, np.imag(ys), linewidth=3.0, label=imgL)
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')


def makeJustAnother2DgraphComplexALLS(xs, yss, fn, labl, xaxisL=[1,14]):
    nstates = yss.shape[0]
    transp = False
    my_dpi = 150
    ratio = (16, 9)
    fig, ax1 = plt.subplots(figsize=ratio)
    ax1.set_xlim(xaxisL)
    ax1.set_ylim([-0.3,2.3])
    for i in range(nstates):
        thing=yss[i]
        rea = np.real(thing)+(i/2)
        ima = np.imag(thing)+(i/2)
        abz = gf.abs2(thing*2)+(i/2)
        plt.plot(xs, rea,linewidth=0.5,ls='--')
        plt.plot(xs, ima,linewidth=0.5,ls='--')
        plt.plot(xs, abz,linewidth=2.0,label=str(i),ls='-')
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')


def makeJustAnother2DgraphComplexSINGLE(xs, ys, fn, labl, xaxisL):
    transp = False
    my_dpi = 150
    ratio = (16, 9)
    imgL = labl + " Imag"
    reaL = labl + " Real"
    sqrL = labl + r" $|\Psi|^2$"
    fig, ax1 = plt.subplots(figsize=ratio)
    ax1.set_ylim([-1,1])
    ax1.set_xlim(xaxisL)
    plt.plot(xs, np.real(ys),linewidth=0.5,label=reaL,ls='--')
    plt.plot(xs, np.imag(ys),linewidth=0.5,label=imgL,ls='--')
    plt.plot(xs, gf.abs2(ys),linewidth=2.0,label=sqrL,ls='-')
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')


def dipoleMatrixGraph1d(inputs, fn, labels, nstates, gridN, distSmall, dipoSmall):
    gridpp      = distSmall.size
    dimensions = len(labels)
    multiplotMtrx = np.empty((dimensions, nstates, nstates, gridpp))
    for i in range(dimensions):
        l1 = labels[i]
        for j in range(nstates):
            l2 = str(j+1)
            for k in range(nstates):
                l3    = str(k+1)
                yaxis = dipoSmall[:,i,j,k]
                name  = fn + l1 + '-' + l2 + 'vs' + l3 + '.png'
                label = l2 + 'vs' + l3
                if inputs.singleDipGrapf:
                    makeJustAnother2Dgraph(distSmall,yaxis,name,label)
                multiplotMtrx[i,j,k] = yaxis
        name  = fn + 'multiplot-' + labels[i] + '.png'
        makeMultiPlotMatrix(distSmall, multiplotMtrx[i], name, "LiHAst", nstates)


def makeMultiPlotMatrix(xs, ys, fn, systemName, nstates):
    '''
    This will make a (nstate,nstate) matrix with changes along R.
    xs      -> array :: np.array
    ys      -> matrix :: np.array (nstates,nstates,gridpoint)
    fn      -> output filename :: String
    nstates -> number of elecrtronic states :: Integer
    '''
    transp     = False
    my_dpi     = 250
    ratio      = (16, 9)
    label_size = 20
    limit      = np.absolute(ys).max()
    f, axarr   = plt.subplots(nstates, nstates, figsize=ratio)
    f.subplots_adjust(hspace=0.0, wspace=0.0)
    f.suptitle(str(limit)+ ' ' + fn)
    statesLab = getLabels(systemName)()[0]
    for i in range(nstates):
        n   = list(reversed(range(nstates)))[i]
        for j in range(nstates):
            plt.setp(axarr[n,j].get_xticklabels(), visible=False)
            plt.setp(axarr[n,j].get_yticklabels(), visible=False)
            axarr[n, j].set_ylim([-limit,limit])
            axarr[n, j].plot(xs,ys[i,j])
            if n == (nstates-1):
               axarr[n, j].set_xlabel(statesLab[j])
            if j == 0:
               axarr[n, j].set_ylabel(statesLab[i])
            axarr[n, j].tick_params(labelsize=label_size)

    #plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=True)
    #plt.setp([a.get_xticklabels() for a in axarr[(nstates-1), :]], visible=True)

    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)


def make2DgraphTranspose(xs, ys, fn, systemName, nstates):
    '''
    Makes a 2d graph with xs and the transpose set of value of ys -> ys[:,i]
    xs      -> array :: np.array
    ys      -> matrix :: np.array (gridpoints,nstates)
    fn      -> output filename :: String
    nstates -> number of elecrtronic states :: Integer
    '''
    transp    = False
    my_dpi    = 250
    ratio     = (16, 9)
    fontsize  = 12
    (statesLab,colorsA) = getLabels(systemName)()
    fig, ax1  = plt.subplots(figsize=ratio)
    ax1.set_xlabel('R')
    ax1.set_ylabel('V(R)')
    #ax1.set_xlim([2,5])
    [plt.plot(xs, ys[:,i], linewidth=3.0,color=colorsA[i],label=statesLab[i]) for i in range(nstates)]
    ax1.legend(loc='upper right')
    plt.rcParams.update({'font.size': fontsize})
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)


def makePulseSinglePointGraph(times, statesArr, pulseArr, imagefn, systemName, nstates):
    '''
    Makes the main graphic of pulse + single geometry integration
    times     -> xaxis :: np.array
    statesArr -> yaxis :: np.array (nstates)
    pulseArr  -> yaxis2 :: np.array (x,y,z) the value of the electromagnetic field
    imagefn   -> filename :: String
    nstates   -> electronic states :: Int
    '''
    transp      = False
    my_dpi      = 250
    ratio       = (16, 9)
    fontsize    = 12
    (statesLab,colorsA) = getLabels(systemName)()
    fig, ax1    = plt.subplots(figsize=ratio)
    ax1.set_xlabel('time (a.u.)')
    ax1.set_ylabel('P(t)')
    for i in range(nstates):
        plt.plot(times,statesArr[:,i],linewidth=3.0,color=colorsA[i],label=statesLab[i])
    ax2         = ax1.twinx()
    # ax2.set_ylabel('E(t)', color='r')
    ax2.set_ylabel('E(t)')
    labelsPulse = ['Pulse X','Pulse Y','Pulse Z']
    stylesPulse = ['--',':','-.']
    ax2.set_ylim([-0.05,0.05])
    [plt.plot(times,pulseArr[:,i],linewidth=1.0, linestyle=stylesPulse[i],label=labelsPulse[i]) for i in range(3)]
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.rcParams.update({'font.size': fontsize})
    plt.savefig(imagefn, bbox_inches='tight', dpi=my_dpi, transparent=transp)


def createHeatMapfromH5(fol,fn):
    '''
    From a folder with N files containing the wavefunction, it creates the time vs population
    heatmaps for any state
    '''
    import glob
    fs = sorted(glob.glob(fol + '/' + fn + '*.h5'))
    times = len(fs)
    wf = h5.retrieve_hdf5_data(fs[0], 'WF')
    (nstates,gridN) = wf.shape
    final = np.empty((times,nstates,gridN), dtype = complex)
    for i in range(times):
          final[i] = h5.retrieve_hdf5_data(fs[i], 'WF')
    fun = np.vectorize(gf.abs2)
    allP = fun(final)
    for i in range(nstates):
        singlestate = allP[:,i,:]
        label = 'POP_' + str(i)
        createSingleHeatmap(singlestate, fol,fn,label)
    print(allP.shape)


def createSingleHeatmap(state,fol,fn,label):
    '''
    Given one eletronic state population evolution over time
    creates a single heatmap
    state :: np.array(time,Population) -> the 2D array
    fol   :: String  -> the output folder
    fn    :: String  -> the output filename
    label :: String  -> The label in the key graph
    '''
    transp      = False
    my_dpi      = 250
    ratio       = (16, 9)
    fig, ax1  = plt.subplots(figsize=ratio)
    im = plt.imshow(state, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    fn2 = fol + '/AAA' + fn + label + 'HeatMap.png'
    plt.savefig(fn2, bbox_inches='tight', dpi=my_dpi, transparent=transp)


def createCoherenceHeatMapfromH5(fol,fn,xs):
    '''
    From a folder with N files containing the wavefunction, it
    creates a heatmap of the decoherence between the states given in the list of tuples xs
    fol :: String
    fn :: String
    xs :: [(i,j)]
    '''
    import glob
    fs = sorted(glob.glob(fol + '/' + fn + '*.h5'))
    times = len(fs)
    wf = h5.retrieve_hdf5_data(fs[0], 'WF')
    (nstates,gridN) = wf.shape
    final = np.empty((times,nstates,gridN), dtype = complex)
    for i in range(times):
        final[i] = h5.retrieve_hdf5_data(fs[i], 'WF')
    for (i,j) in xs:
        coherence = np.apply_along_axis(lambda t: ccmu(*t), 0, np.stack((final[:,i,:],final[:,j,:])))
        label = "COHE_" + str(i) + "_vs_" + str(j)
        createSingleHeatmap(coherence,fol,fn,label)


def ccmu(a,b):
    ''' complex conjugate multiplication '''
    return ((np.conjugate(a))*b).real * 2


def createHistogram(array, fn, binNum=20, rang=None):
    '''
    A general function to quickly plot an Histogram on an array
    array :: np.array rank 1
    fn :: String -> Output filepath
    binNum :: Integer or Sequence of integers -> number of bins
    rang :: (downlimit,uplimit) -> the limits of the histogram
    '''
    my_dpi      = 250
    ratio       = (16, 9)
    fig, ax1 = plt.subplots(figsize=ratio)
    plt.hist(array,bins=binNum, range=rang)
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi)
    plt.close('all')

def makeMultiLineDipoleGraph(xs, yss, label, state):
    ''' now I am tired, but this function should be commented and change
    name '''
    my_dpi      = 250
    ratio       = (16, 9)
    ( _ , nplots) = yss.shape
    fig, ax1 = plt.subplots(figsize=ratio)

    for i in range(nplots):
        lbl = str(i+1)
        if state != i :
            plt.plot(xs,yss[:,i], label=lbl)
    colormap = plt.cm.gist_ncar
    colors = [colormap(i) for i in np.linspace(0.1, 0.9,len(ax1.lines))]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])
    ax1.legend(loc='upper right')
    plt.savefig(label, bbox_inches='tight', dpi=my_dpi)

if __name__ == "__main__":
    a = np.arange(11)
    b = np.random.rand(11,9)
    makJusAno2DgrMultiline(a,b,'porchii', 3)


