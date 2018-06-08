''' This module precompute along a 3x3 grid the Jacobian to use in the podolsky form of the Kinetic energy operator'''

import numpy as np
from numpy import (dot,cos,sin,sqrt,array,deg2rad,stack)
from quantumpropagator import good
#from numpy import (dot,cos,sin,sqrt,sum,array,stack,deg2rad)
#from numpy.linalg import norm



def calc_s_mat(phi,gam,the,verbose=None):
    '''
    phi :: Double - values of phi
    gam :: Double - values of gam in RADIANS
    the :: Double - values of the in RADIANS
    '''
    verbose = verbose or False
    ang2boh = 1.889725988
    umass = 1836

    # 
    # In this part we want to take out masses from the calculations of these coefficients
    # because we just want to compute the momentum.
    # 

    cc    = 1.541 * ang2boh
    ch    = 1.541 * ang2boh

    # derivatives of phi
    dp_c8x  = (-0.165777 * ang2boh / 6)
    dp_c8y  = (0.067387  * ang2boh / 6)
    dp_c8z  = (0.016393  * ang2boh / 6)
    dp_c9x  = (-0.145170 * ang2boh / 6)
    dp_c9y  = (-0.096085 * ang2boh / 6)
    dp_c9z  = (-0.143594 * ang2boh / 6)
    dp_h12x = (-0.520977 * ang2boh / 6)
    dp_h12y = (0.086124  * ang2boh / 6)
    dp_h12z = (0.316644  * ang2boh / 6)
    dp_h13x = (0.450303  * ang2boh / 6)
    dp_h13y = (-0.048000 * ang2boh / 6)
    dp_h13z = (0.245432  * ang2boh / 6)

    dp_phi  = array([ dp_c8x,   dp_c8y,  dp_c8z,
                      dp_c9x,   dp_c9y,  dp_c9z,
                     -dp_c8x,  -dp_c8y,  dp_c8z,
                     -dp_c9x,  -dp_c9y,  dp_c9z,
                      dp_h12x,  dp_h12y, dp_h12z,
                      dp_h13x,  dp_h13y, dp_h13z,
                     -dp_h12x, -dp_h12y, dp_h12z,
                     -dp_h13x, -dp_h13y, dp_h13z])

    # derivatives of gam
    dg_c8x  = cc * sin(the) * sin(gam)
    dg_c8y  = -cc * cos(gam)
    dg_c8z  = cc * cos(the) * sin(gam)
    dg_c9x  = -cc * sin(the) * sin(gam)
    dg_c9y  = -cc * cos(gam)
    dg_c9z  = cc * cos(the) * sin(gam)
    dg_h12x = ch * sin(the) * sin(gam)
    dg_h12y = -ch * cos(gam)
    dg_h12z = ch * cos(the) * sin(gam)
    dg_h13x = -ch * sin(the) * sin(gam)
    dg_h13y = -ch * cos(gam)
    dg_h13z = ch * cos(the) * sin(gam)
    dg_gam  = array([ dg_c8x,   dg_c8y,  dg_c8z,
                      dg_c9x,   dg_c9y,  dg_c9z,
                     -dg_c8x,  -dg_c8y,  dg_c8z,
                     -dg_c9x,  -dg_c9y,  dg_c9z,
                      dg_h12x,  dg_h12y, dg_h12z,
                      dg_h13x,  dg_h13y, dg_h13z,
                     -dg_h12x, -dg_h12y, dg_h12z,
                     -dg_h13x, -dg_h13y, dg_h13z])

    # derivatives of the
    dt_c8x  = -cc * cos(the) * cos(gam)
    dt_c8y  =  0
    dt_c8z  =  cc * sin(the) * cos(gam)
    dt_c9x  =  cc * cos(the) * cos(gam)
    dt_c9y  =  0
    dt_c9z  =  cc * sin(the) * cos(gam)
    dt_h12x = -ch * cos(the) * cos(gam)
    dt_h12y =  0
    dt_h12z =  ch * sin(the) * cos(gam)
    dt_h13x =  ch * cos(the) * cos(gam)
    dt_h13y =  0
    dt_h13z =  ch * sin(the) * cos(gam)
    dt_the  = array([ dt_c8x,   dt_c8y,  dt_c8z,
                      dt_c9x,   dt_c9y,  dt_c9z,
                     -dt_c8x,  -dt_c8y,  dt_c8z,
                     -dt_c9x,  -dt_c9y,  dt_c9z,
                      dt_h12x,  dt_h12y, dt_h12z,
                      dt_h13x,  dt_h13y, dt_h13z,
                     -dt_h12x, -dt_h12y, dt_h12z,
                     -dt_h13x, -dt_h13y, dt_h13z])

    # matrix derivative_mat is cartesian by internal, but we want internal by cartesian, thus we
    # invert. This give the LEFT inverse.
    dcart_dint_mat = stack((dp_phi,dg_gam,dt_the),axis=1)
    dint_dcart_mat = np.linalg.pinv(dcart_dint_mat)
    identity = dint_dcart_mat @ dcart_dint_mat # this must be identity matrix, of course (a@b does NOT)

    good('New Run')

    if verbose:
        stringZ = '\nCart/int matrix:\n{}\nShape: {}\n\nS (inverted) Matrix:\n{}\n{}'
        print(stringZ.format(dcart_dint_mat, dcart_dint_mat.shape, dint_dcart_mat, dint_dcart_mat.shape))

    # good('only Z')

    # only_z_mat = dcart_dint_mat[2:24:3] # take only z values
    # only_z_inverse = np.linalg.pinv(only_z_mat)
    # identity2 = only_z_inverse @ only_z_mat

    # if verbose:
    #     stringZ = '\n\n\nCart/int matrix Z :\n{}\nShape: {}\n\nS (Z inverted) Matrix:\n{}\n{}'
    #     print(stringZ.format(only_z_mat, only_z_mat.shape, only_z_inverse,only_z_inverse.shape))
    #     print (identity2)

    return(dint_dcart_mat)


def calc_g_G(phi,gam,the,verbose=None):
    '''
    phi :: Double - values of phi
    gam :: Double - values of gam in RADIANS
    the :: Double - values of the in RADIANS
    '''
    verbose = verbose or False
    ang2boh = 1.889725988
    umass = 1836
    phiNumber = 0.06
    # I already controlled that the lack of parentheses gives the result I want
    # Hydrogen has the mass to 1 but it is the C length that controls the
    # derivative
    cc    = 1.541 * ang2boh * sqrt(12 * umass)
    ch    = 1.541 * ang2boh * sqrt(1  * umass)

    # derivatives of phi
    dp_c8x  = (-0.165777 * ang2boh / phiNumber) * sqrt(12 * umass)
    dp_c8y  = (0.067387  * ang2boh / phiNumber) * sqrt(12 * umass)
    dp_c8z  = (0.016393  * ang2boh / phiNumber) * sqrt(12 * umass)
    dp_c9x  = (-0.145170 * ang2boh / phiNumber) * sqrt(12 * umass)
    dp_c9y  = (-0.096085 * ang2boh / phiNumber) * sqrt(12 * umass)
    dp_c9z  = (-0.143594 * ang2boh / phiNumber) * sqrt(12 * umass)
    dp_h12x = (-0.520977 * ang2boh / phiNumber) * sqrt(1  * umass)
    dp_h12y = (0.086124  * ang2boh / phiNumber) * sqrt(1  * umass)
    dp_h12z = (0.316644  * ang2boh / phiNumber) * sqrt(1  * umass)
    dp_h13x = (0.450303  * ang2boh / phiNumber) * sqrt(1  * umass)
    dp_h13y = (-0.048000 * ang2boh / phiNumber) * sqrt(1  * umass)
    dp_h13z = (0.245432  * ang2boh / phiNumber) * sqrt(1  * umass)
    dp_phi  = array([ dp_c8x,   dp_c8y,  dp_c8z,
                      dp_c9x,   dp_c9y,  dp_c9z,
                     -dp_c8x,  -dp_c8y,  dp_c8z,
                     -dp_c9x,  -dp_c9y,  dp_c9z,
                      dp_h12x,  dp_h12y, dp_h12z,
                      dp_h13x,  dp_h13y, dp_h13z,
                     -dp_h12x, -dp_h12y, dp_h12z,
                     -dp_h13x, -dp_h13y, dp_h13z])

    # derivatives of gam
    dg_c8x  = cc * sin(the) * sin(gam)
    dg_c8y  = -cc * cos(gam)
    dg_c8z  = cc * cos(the) * sin(gam)
    dg_c9x  = -cc * sin(the) * sin(gam)
    dg_c9y  = -cc * cos(gam)
    dg_c9z  = cc * cos(the) * sin(gam)
    dg_h12x = ch * sin(the) * sin(gam)
    dg_h12y = -ch * cos(gam)
    dg_h12z = ch * cos(the) * sin(gam)
    dg_h13x = -ch * sin(the) * sin(gam)
    dg_h13y = -ch * cos(gam)
    dg_h13z = ch * cos(the) * sin(gam)
    dg_gam  = array([ dg_c8x,   dg_c8y,  dg_c8z,
                      dg_c9x,   dg_c9y,  dg_c9z,
                     -dg_c8x,  -dg_c8y,  dg_c8z,
                     -dg_c9x,  -dg_c9y,  dg_c9z,
                      dg_h12x,  dg_h12y, dg_h12z,
                      dg_h13x,  dg_h13y, dg_h13z,
                     -dg_h12x, -dg_h12y, dg_h12z,
                     -dg_h13x, -dg_h13y, dg_h13z])

    # derivatives of the
    dt_c8x  = -cc * cos(the) * cos(gam)
    dt_c8y  =  0
    dt_c8z  =  cc * sin(the) * cos(gam)
    dt_c9x  =  cc * cos(the) * cos(gam)
    dt_c9y  =  0
    dt_c9z  =  cc * sin(the) * cos(gam)
    dt_h12x = -ch * cos(the) * cos(gam)
    dt_h12y =  0
    dt_h12z =  ch * sin(the) * cos(gam)
    dt_h13x =  ch * cos(the) * cos(gam)
    dt_h13y =  0
    dt_h13z =  ch * sin(the) * cos(gam)
    dt_the  = array([ dt_c8x,   dt_c8y,  dt_c8z,
                      dt_c9x,   dt_c9y,  dt_c9z,
                     -dt_c8x,  -dt_c8y,  dt_c8z,
                     -dt_c9x,  -dt_c9y,  dt_c9z,
                      dt_h12x,  dt_h12y, dt_h12z,
                      dt_h13x,  dt_h13y, dt_h13z,
                     -dt_h12x, -dt_h12y, dt_h12z,
                     -dt_h13x, -dt_h13y, dt_h13z])

    # g matrix
    g_11 = dot(dp_phi,dp_phi)
    g_22 = dot(dg_gam,dg_gam)
    g_33 = dot(dt_the,dt_the)
    g_12 = dot(dp_phi,dg_gam)
    g_13 = dot(dp_phi,dt_the)
    det_g = g_11 * g_22 * g_33 - g_12**2 * g_33 + g_13**2 * g_22
    if verbose:
        print('\nDet:\n{}'.format(det_g))

    # G' matrix
    G_pp = (g_33 * g_22)
    G_pg = (-g_33 * g_12)
    G_gp = G_pg
    G_pt = (-g_22 * g_13)
    G_tp = G_pt
    G_gg = (g_33 * g_11 - g_13**2)
    G_gt = (g_13 * g_12)
    G_tg = G_gt
    G_tt = (g_22 * g_11 - g_12**2)
    if verbose:
        print('\nG elements:\n{} {} {} {} {} {}'.format(G_pp,G_pg,G_pt,G_gg,G_gt,G_tt))

    # derivatives of det(g) they're 6
    dgdet_g = 32 * (cc**2 + ch**2) * cos(gam) * (-2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 +dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) * sin(gam) + sin(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 + cos(gam) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(gam) + cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the))) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the))) + sin(gam) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the)))**2)
    dtdet_g = -32 * (cc**2 + ch**2) * cos(gam)**3 * ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(gam) + cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the)))
    dgdgdet_g = 32 * (cc**2 + ch**2) * (-2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 + dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) * cos(gam)**2 + (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y))**2 * cos(gam)**4 + 2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 + dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) * sin(gam)**2 - 5 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y))**2 * cos(gam)**2 * sin(gam)**2 - 4 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam)**3 * sin(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the)) - cos(gam)**4 * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the))**2 + 5 * cos(gam)**2 * sin(gam)**2 * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the))**2 + 4 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam)**3 * sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * sin(the)) - 4 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) * sin(gam)**3 * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * sin(the)) + cos(gam)**2 * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 - sin(gam)**2 * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 + cos(gam)**2 * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the)))**2 - sin(gam)**2 * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the)))**2)
    dtdgdet_g = 32 * (cc**2 + ch**2) * cos(gam)**2 * ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * (-(cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * (-1 + 2 * cos(2 * gam)) + 2 * (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) * sin(2 * gam) + 2 * (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(2 * gam) * sin(the))
    dgdtdet_g = dtdgdet_g
    dtdtdet_g = 32 * (cc**2 + ch**2) * cos(gam)**3 * (cos(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 - ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the)) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(gam) + cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the))))
    if verbose:
        print('\nDerivatives g:\n{} {} {} {} {} {}'.format(dtdet_g,dgdet_g,dgdgdet_g,dtdtdet_g,dtdgdet_g,dgdtdet_g))

    # Now the G' elements. They're 6
    dgGggs = 8 * cos(gam) * sin(gam) * (-2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 +dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) + ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2)
    dgGgps = 2 * (cc**2 +ch**2) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * (cos(gam) + 3 * cos(3 * gam)) * cos(the) - 12 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam)**2 * sin(gam) - (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * (cos(gam) + 3 * cos(3 * gam)) * sin(the))
    dgGgts = 4 * ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(2 * gam) + cos(2 * gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the)))
    dtGtgs = 2 * cos(gam) * (-2 * sin(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 + ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the)) * (-2 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + 2 * sin(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the))))
    dtGtps = -8 * (cc**2 + ch**2) * cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the))
    dtGtts = -8 * sin(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * (-(cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the)))

    if verbose:
        print('\nDerivatives G:\n{} {} {} {} {} {}'.format(dgGggs,dgGgps,dgGgts,dtGtgs,dtGtps,dtGtts))

    # Now it is time to write down the effective coefficients of T. Labels are 0,1,2 (indicating derivatives) and the usual p t g
    hbar = 1
    hb = hbar**2

    # zero order derivative coefficients
    Tgg0 = (hb * dgGggs * dgdet_g)/(8 * det_g**2) - (7 * hb * G_gg * dgdet_g**2)/(32 * det_g**3) + (hb * G_gg * dgdgdet_g)/(8 * det_g**2)
    Ttt0 = (hb * dtGtts * dtdet_g)/(8 * det_g**2) - (7 * hb * G_tt * dtdet_g**2)/(32 * det_g**3) + (hb * G_tt * dtdtdet_g)/(8 * det_g**2)
    Tgt0 = (hb * dgGgts * dtdet_g)/(8 * det_g**2) - (7 * hb * G_gt * dgdet_g * dtdet_g)/(32 * det_g**3) + (hb * G_gt * dgdtdet_g)/(8 * det_g**2)
    Ttg0 = (hb * dtGtgs * dgdet_g)/(8 * det_g**2) - (7 * hb * G_tg * dgdet_g * dtdet_g)/(32 * det_g**3) + (hb * G_tg * dtdgdet_g)/(8 * det_g**2)
    Tgp0 = 0
    Ttp0 = 0
    Tpg0 = 0
    Tpt0 = 0
    Tpp0 = 0

    # first order derivative coefficients -> the cross terms 3/8 + 1/8 adds up to 1/2! So even if in the latex formulas they appear in two
    # different term, we pack them togheter in this calculation, so the Ttot matrix is still 3x9!
    Tgg1 = - (hb * dgGggs)/(2 * det_g) + (    hb * G_gg * dgdet_g)/(2 * det_g**2)
    Ttt1 = - (hb * dtGtts)/(2 * det_g) + (    hb * G_tt * dtdet_g)/(2 * det_g**2)
    Tgt1 = - (hb * dgGgts)/(2 * det_g) + (    hb * G_gt * dgdet_g)/(2 * det_g**2)
    Ttg1 = - (hb * dtGtgs)/(2 * det_g) + (    hb * G_tg * dtdet_g)/(2 * det_g**2)
    Tgp1 = - (hb * dgGgps)/(2 * det_g) + (3 * hb * G_gp * dgdet_g)/(8 * det_g**2)
    Ttp1 = - (hb * dtGtps)/(2 * det_g) + (3 * hb * G_tp * dtdet_g)/(8 * det_g**2)
    Tpg1 =                             + (3 * hb * G_pg * dgdet_g)/(8 * det_g**2)
    Tpt1 =                             + (3 * hb * G_pt * dtdet_g)/(8 * det_g**2)
    Tpp1 = 0

    # second order derivative coefficients
    Tgg2 = - (hb * G_gg)/(2 * det_g)
    Ttt2 = - (hb * G_tt)/(2 * det_g)
    Tgt2 = - (hb * G_gt)/(2 * det_g)
    Ttg2 = - (hb * G_tg)/(2 * det_g)
    Tgp2 = - (hb * G_gp)/(2 * det_g)
    Ttp2 = - (hb * G_tp)/(2 * det_g)
    Tpg2 = - (hb * G_pg)/(2 * det_g)
    Tpt2 = - (hb * G_pt)/(2 * det_g)
    Tpp2 = - (hb * G_pp)/(2 * det_g)

    # until here the order is the same as the Latex document (from hard gg to easy pp).
    # But here we need to reorder as it is in the main matrix
    Ttot = array([[Tpp0,Tpp1,Tpp2],
                  [Tpg0,Tpg1,Tpg2],
                  [Tpt0,Tpt1,Tpt2],
                  [Tgp0,Tgp1,Tgp2],
                  [Tgg0,Tgg1,Tgg2],
                  [Tgt0,Tgt1,Tgt2],
                  [Ttp0,Ttp1,Ttp2],
                  [Ttg0,Ttg1,Ttg2],
                  [Ttt0,Ttt1,Ttt2]])
    if verbose:
        print('\nT coefficient Matrix:\n',Ttot)
    return(Ttot)

if __name__ == "__main__":
    #calc_s_mat(2.0,15,50,True)
    calc_g_G(2,15,50,True)
