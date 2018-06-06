import glob
import quantumpropagator as qp
import numpy as np
from mayavi.mlab import *



def fourdThing(fn):
    '''
    we try myavi
    '''
    a = qp.retrieve_hdf5_data(fn,'WF')[:,:,:,0]
    time = qp.retrieve_hdf5_data(fn,'Time')[0]
    fig = figure(size=(600,600),fgcolor=(0, 0, 0), bgcolor=(1, 1, 1))
    contour3d(qp.abs2(a),contours=10, transparent=True, opacity=0.8)
    #title('Time - {:5.2f}'.format(time))
    xlabel('phi')
    ylabel('gam')
    zlabel('the')
    #view(132.05741302902214, 62.73780301264113, 187.6797494627067, np.array([ 15.00000048, 11., 1087643]))
    view(47.01, 79.90, 105.94, np.array([ 15.00, 11.00, 15.00]))
    savefig(fn + '.a.png')

    view(0.0, 90.0, 87.034, np.array([ 15.,  11.,  15.]))
    savefig(fn + '.b.png')

    view(90.0, 90.0, 87.034, np.array([ 15.,  11.,  15.]))
    savefig(fn + '.c.png')

    view(0.0, 0.0, 87.034, np.array([ 15.,  11.,  15.]))
    savefig(fn + '.d.png')
    close()


def main():
    fol = '/home/alessio/Desktop/a-3dScanSashaSupport/n-Propagation/results/aaa_input_FinerGrid_0031'
    filelist = sorted(glob.glob(fol + '/Gau*.h5'))
    for i in range(len(filelist)):
    #for i in range(10):
        fourdThing(filelist[i])


if __name__ == "__main__":
    main()

