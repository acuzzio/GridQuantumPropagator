'''
This is the module where the program generates graphics
'''

import numpy as np
#import matplotlib as mpl
#mpl.use('Agg')
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

def makeJustAnother2DgraphMULTI(xs,yss,fn,labl,lw=None):
    '''
    xs   :: np.array[Double]         <- domain
    yss  :: np.array[Double,Double]  <- 2d Array (multiple value at each X)
    fn   :: FilePath                 <- output name
    labl :: String - the name on the key, that will be incremented
    '''
    lw = lw or 2.0
    transp    = False
    my_dpi    = 150
    ratio     = (16, 9)
    _, ax1  = plt.subplots(figsize=ratio)
    (_,nstates) = yss.shape
    for ind in np.arange(nstates):
        labThis = labl + " " + str(ind)
        plt.plot(xs, yss[:,ind],linewidth=lw,label=labThis)
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')


def makeJustAnother2Dgraph(fn,labl,ys,xs=None):
    '''
    xs :: np.array[Double]
    yx :: np.array[Double]
    fn :: FilePath
    labl :: String - the name on the key
    '''
    length = ys.size
    xs = xs or np.arange(length)
    transp    = False
    my_dpi    = 150
    ratio     = (16, 9)
    _, ax1  = plt.subplots(figsize=ratio)
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
    _ , ax1 = plt.subplots(figsize=ratio)
    ax1.set_ylabel('E A.U. (Hartree)')
    ax1.set_xlabel('Time A.U.')
    plt.plot(timesArray, gaus, linewidth=2.0, label="Envelope")
    plt.plot(timesArray, puls, linewidth=2.0, label="Pulse")
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')

def heatMap2dWavefunction(wf,name,time,vmaxV=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    fig = plt.figure(figsize=(15, 8), dpi= 80, facecolor='w', edgecolor='k')
    plt.title('Time = {:10.5f} fs'.format(time))
    plt.ylabel('Gamma')
    plt.xlabel('Theta')

    # this is to get a nice colorbar on the side
    ax = plt.gca()
    vmaxV = vmaxV or 0.1
    im = ax.imshow(gf.abs2(wf), cmap='hot', vmax=vmaxV)
    #im = ax.imshow(qp.abs2(wf), cmap='PuBu_r', vmax=0.4)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    fig.savefig(name)
    plt.close()

def makeJustAnother2DgraphComplex(xs,ys,fn,labl,xlimit=None):
    transp = False
    xlimit = xlimit or [-1,1]
    my_dpi = 150
    ratio = (16, 9)
    imgL = labl + " Imag"
    reaL = labl + " Real"
    fig, ax1 = plt.subplots(figsize=ratio)
    ax1.set_ylim(xlimit)
    plt.plot(xs, np.real(ys), linewidth=3.0, label=reaL)
    plt.plot(xs, np.imag(ys), linewidth=3.0, label=imgL)
    ax1.legend(loc='upper right')
    plt.savefig(fn, bbox_inches='tight', dpi=my_dpi, transparent=transp)
    plt.close('all')


def makeJustAnother2DgraphComplexALLS(xs, yss, fn, labl, xaxisL=None):
    xaxisL = xaxisL or [1,14]
    nstates = yss.shape[0]
    transp = False
    my_dpi = 150
    ratio = (16, 9)
    fig, ax1 = plt.subplots(figsize=ratio)
    ax1.set_xlim(xaxisL)
    ax1.set_ylim([-4,4])
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


def makeJustAnother2DgraphComplexSINGLE(xs, ys, fn, labl):
    transp = False
    my_dpi = 150
    ratio = (16, 9)
    imgL = labl + " Imag"
    reaL = labl + " Real"
    sqrL = labl + r" $|\Psi|^2$"
    fig, ax1 = plt.subplots(figsize=ratio)
    ax1.set_ylim([-1,1])
    #ax1.set_xlim(xaxisL)
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
    for i in range(nstates):
        plt.plot(xs, ys[:,i], linewidth=3.0,color=colorsA[i],label=statesLab[i])
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
    for i in range(3):
        plt.plot(times,pulseArr[:,i],linewidth=1.0, linestyle=stylesPulse[i],label=labelsPulse[i])
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
    plt.close('all')

if __name__ == "__main__":
    a = np.arange(11)
    b = np.random.rand(11,9)
    makJusAno2DgrMultiline(a,b,'porchii', 3)


def mathematicaListGenerator(a,b,c):
    '''
    This function takes a,b,c and printout a file that should be copied/pasted
    into Mathematica 11.00
    a :: np.array(ngrid) <- first axis
    b :: np.array(ngrid) <- second axis
    c :: np.array(ngrid,second) <- actual values
    second should be nstates or whatever dimension the data in packed.
    to print vectors like this, every single point needs its coordinates, so we
    have to supply every point like (x,y,z) triplet.
    xclip is the best
    '''
    import string
    letter = list(string.ascii_lowercase)
    #(length, ) = c.shape
    #surfaces = 1
    (length, surfaces) = c.shape
    finalString = 'ListPlot3D[{'
    matheString = ''
    for sur in np.arange(surfaces):
        fName = letter[sur]
        if ((sur != surfaces-1)):
            finalString = finalString + fName + ','
        else:
            finalString = finalString + fName + '}]'
        stringF = fName + ' = {'
        for ind in np.arange(length):
            first = str(a[ind]) + ','
            second = str(b[ind]) + ','
            third = '{:16.14f}'.format(c[ind,sur])
            if (ind != length-1):
                stringF = stringF+"{"+first+second+third+'},'
            else:
                stringF = stringF+"{"+first+second+third+'}}'
        matheString = matheString + "\n" + stringF
    #print(matheString)
    fn = 'hugeVector'
    with open(fn, "w") as myfile:
        myfile.write(matheString)
    print('\ncat ' + fn + ' | xclip -selection clipboard')
    print('\n'+finalString+'\n')

def gnuSplotCircle(a,b,c):
    '''
    to generate data and script for gnuplot to generate
    3d plot function of circles around conical intersection
    a :: np.array(ngrid) <- first axis
    b :: np.array(ngrid) <- second axis
    c :: np.array(ngrid,second) <- actual values (energies)

    I wanted to do this gnuSplot as a general function, but gnuplot is peculiar and
    I needed to adjust the datafile with some "case dependent" actions.
    To paint good circular data, gnuplot needs a SPACE between different
    circles and THE FIRST LINE OF EACH CIRCLE repeated (to close it).
    This is done knowing that the first point will always be at 0 degree, thus
    checking for cosine > 0.0 and sin = 0.0. This is why the madness firstLine
    fullString, etc. etc.
    '''
    (length, surfaces) = c.shape
    fn = 'dataFile'
    fnS = 'GnuplotScript'
    with open(fn, 'w') as outF:
        firstLine = ''
        for i in range(length):
            strC = ' '.join(map(str, c[i]))
            # I generate a file with last column 1 to... EXAGERATE Z AXIS if
            # you wish... wow... I should comment more...
            fullString = "{:3.4f} {:3.4f} {} 1\n".format(a[i],b[i],strC)
            if a[i] == 0.0 and b[i] > 0.0:
                if firstLine == '':
                    firstLine = fullString
                else:
                    firstNow = firstLine
                    firstLine = fullString
                    fullString = firstNow + '\n' + fullString
            outF.write(fullString)
        outF.write(firstLine)
    with open(fnS, 'w') as scri:
        header = '''#set dgrid3d
#set pm3d
#set style data lines
set xlabel "C1-C5"
set ylabel "C2-C5"
set ticslevel 0
splot '''
        scri.write(header)
        for i in range(surfaces):
            iS = str(i)
            i1 = str(i + 1)
            i3 = str(i + 3)
            cc3 = str(surfaces+3)
            string='"'+fn+'" u 1:2:'+i3+':($'+cc3+'+'+iS+') t "S_{'+iS+'}'+i1+'" w l'
            if (i != surfaces-1):
                string = string + ', '
            scri.write(string)

