''' This module precompute along a 3x3 grid the Jacobian to use in the podolsky form of the Kinetic energy operator'''

from numpy import (dot,cos,sin,sqrt,array,stack,deg2rad)
#from numpy import (dot,cos,sin,sqrt,sum,array,stack,deg2rad)
#from numpy.linalg import norm

def calc_g_G(phi,gam,the):
    gam = deg2rad(gam)
    the = deg2rad(the)
    ang2boh = 1.889725988
    umass = 1836

    # I already controlled that the lack of parentheses gives the result I want
    # Hydrogen has the mass to 1 but it is the C length that controls the
    # derivative
    cc    = 1.541 * ang2boh * sqrt(12 * umass)
    ch    = 1.541 * ang2boh * sqrt(1  * umass)

    # derivatives of phi
    dp_c8x  = -0.165777 * ang2boh / 6 * sqrt(12 * umass)
    dp_c8y  = 0.067387  * ang2boh / 6 * sqrt(12 * umass)
    dp_c8z  = 0.016393  * ang2boh / 6 * sqrt(12 * umass)
    dp_c9x  = -0.145170 * ang2boh / 6 * sqrt(12 * umass)
    dp_c9y  = -0.096085 * ang2boh / 6 * sqrt(12 * umass)
    dp_c9z  = -0.143594 * ang2boh / 6 * sqrt(12 * umass)
    dp_h12x = -0.520977 * ang2boh / 6 * sqrt(1  * umass)
    dp_h12y = 0.086124  * ang2boh / 6 * sqrt(1  * umass)
    dp_h12z = 0.316644  * ang2boh / 6 * sqrt(1  * umass)
    dp_h13x = 0.450303  * ang2boh / 6 * sqrt(1  * umass)
    dp_h13y = -0.048000 * ang2boh / 6 * sqrt(1  * umass)
    dp_h13z = 0.245432  * ang2boh / 6 * sqrt(1  * umass)
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
    #print('\nDet:\n{}'.format(det_g))

    # G matrix
    G_pp = (g_33 * g_22)/det_g
    G_pg = (-g_33 * g_12)/det_g
    G_gp = G_pg
    G_pt = (-g_22 * g_13)/det_g
    G_tp = G_pt
    G_gg = (g_33 * g_11 - g_13**2)/det_g
    G_gt = (g_13 * g_12)/det_g
    G_tg = G_gt
    G_tt = (g_22 * g_11 - g_12**2)/det_g
    # print('\nG elements:\n{} {} {} {} {} {}'.format(G_pp,G_pg,G_pt,G_gg,G_gt,G_tt))

    # derivatives of det(g) they're 6
    dgdet_g = 32 * (cc**2 + ch**2) * cos(gam) * (-2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 +dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) * sin(gam) + sin(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 + cos(gam) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(gam) + cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the))) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the))) + sin(gam) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the)))**2)
    dtdet_g = -32 * (cc**2 + ch**2) * cos(gam)**3 * ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(gam) + cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the)))
    dgdgdet_g = 32 * (cc**2 + ch**2) * (-2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 + dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) * cos(gam)**2 + (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y))**2 * cos(gam)**4 + 2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 + dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) * sin(gam)**2 - 5 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y))**2 * cos(gam)**2 * sin(gam)**2 - 4 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam)**3 * sin(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the)) - cos(gam)**4 * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the))**2 + 5 * cos(gam)**2 * sin(gam)**2 * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the))**2 + 4 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam)**3 * sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * sin(the)) - 4 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) * sin(gam)**3 * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * sin(the)) + cos(gam)**2 * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 - sin(gam)**2 * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 + cos(gam)**2 * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the)))**2 - sin(gam)**2 * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (-cc * dp_c8x + cc * dp_c9x - ch * dp_h12x + ch * dp_h13x) * sin(the)))**2)
    dtdgdet_g = 32 * (cc**2 + ch**2) * cos(gam)**2 * ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * (-(cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * (-1 + 2 * cos(2 * gam)) + 2 * (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) * sin(2 * gam) + 2 * (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(2 * gam) * sin(the))
    dgdtdet_g = dtdgdet_g
    dtdtdet_g = 32 * (cc**2 + ch**2) * cos(gam)**3 * (cos(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 - ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the)) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(gam) + cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the))))

    # print('\nDerivatives g:\n{} {} {} {} {} {}'.format(dtdet_g,dgdet_g,dgdgdet_g,dtdtdet_g,dtdgdet_g,dgdtdet_g))

    # Now the G' elements. They're 6
    dgGggs = 8 * cos(gam) * sin(gam) * (-2 * (cc**2 + ch**2) * (dp_c8x**2 + dp_c8y**2 + dp_c8z**2 + dp_c9x**2 + dp_c9y**2 +dp_c9z**2 + dp_h12x**2 + dp_h12y**2 + dp_h12z**2 + dp_h13x**2 + dp_h13y**2 + dp_h13z**2) + ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2)
    dgGgps = 2 * (cc**2 +ch**2) * (-(cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * (cos(gam) + 3 * cos(3 * gam)) * cos(the) - 12 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam)**2 * sin(gam) - (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * (cos(gam) + 3 * cos(3 * gam)) * sin(the))
    dgGgts = 4 * ((cc * (-dp_c8x + dp_c9x) + ch * (-dp_h12x + dp_h13x)) * cos(the) + (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * ((cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * sin(2 * gam) + cos(2 * gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the)))
    dtGtgs = 2 * cos(gam) * (-2 * sin(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the))**2 + ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the)) * (-2 * (cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + 2 * sin(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the))))
    dtGtps = -8 * (cc**2 + ch**2) * cos(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * sin(the))
    dtGtts = -8 * sin(gam) * ((cc * (dp_c8x - dp_c9x) + ch * (dp_h12x - dp_h13x)) * cos(the) - (cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * sin(the)) * (-(cc * (dp_c8y + dp_c9y) + ch * (dp_h12y + dp_h13y)) * cos(gam) + sin(gam) * ((cc * (dp_c8z + dp_c9z) + ch * (dp_h12z + dp_h13z)) * cos(the) + (cc * dp_c8x - cc * dp_c9x + ch * dp_h12x - ch * dp_h13x) * sin(the)))
    #print('\nDerivatives G:\n{} {} {} {} {} {}'.format(dgGggs,dgGgps,dgGgts,dtGtgs,dtGtps,dtGtts))

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

    # first order derivative coefficients
    Tgg1 = - (hb * dgGggs)/(2 * det_g) + (3 * hb * G_gg * dgdet_g)/(8 * det_g**2)
    Ttt1 = - (hb * dtGtts)/(2 * det_g) + (3 * hb * G_tt * dtdet_g)/(8 * det_g**2)
    Tgt1 = - (hb * dgGgts)/(2 * det_g) + (3 * hb * G_gt * dgdet_g)/(8 * det_g**2)
    Ttg1 = - (hb * dtGtgs)/(2 * det_g) + (3 * hb * G_tg * dtdet_g)/(8 * det_g**2)
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

    Ttot = array([[Tpp0,Tpp1,Tpp2],
                  [Tpg0,Tpg1,Tpg2],
                  [Tpt0,Tpt1,Tpt2],
                  [Tgp0,Tgp1,Tgp2],
                  [Tgg0,Tgg1,Tgg2],
                  [Tgt0,Tgt1,Tgt2],
                  [Ttp0,Ttp1,Ttp2],
                  [Ttg0,Ttg1,Ttg2],
                  [Ttt0,Ttt1,Ttt2]])
    print('\nT coefficient Matrix:\n',Ttot)



def all_of_them():
    thetas = [130.000,129.101,128.202,127.303,126.404,125.506,124.607,123.708,122.809,121.910,121.011,120.112,119.213,118.315,117.416,116.517,115.618,114.719,113.820,112.921,112.022,111.124,110.225,109.326,108.427,107.528,106.629,105.730,104.831,103.933,103.034,102.135,101.236,100.337,099.438,098.539,097.640,096.742,095.843,094.944,094.045,093.146,092.247,091.348,090.449,089.551,088.652,087.753,086.854,085.955,085.056,084.157,083.258,082.360,081.461,080.562,079.663,078.764,077.865,076.966,076.067,075.169,074.270,073.371,072.472,071.573,070.674,069.775,068.876,067.978,067.079,066.180,065.281,064.382,063.483,062.584,061.685,060.787,059.888,058.989,058.090,057.191,056.292,055.393,054.494,053.596,052.697,051.798,050.899,050.000]
    gammas = [008.000,008.632,009.263,009.895,010.526,011.158,011.789,012.421,013.053,013.684,014.316,014.947,015.579,016.211,016.842,017.474,018.105,018.737,019.368,020.000]
    phis = [010.000,009.000,008.000,007.000,006.000,005.000,004.000,003.000,002.000,001.000,000.000,-001.000,-002.000,-003.000,-004.000,-005.000,-006.000,-007.000,-008.000,-009.000,-010.000]
    for the in thetas:
        for gam in gammas:
            for phi in phis:
                print(the,gam,phi)
                calc_g_G(phi,gam,the/2)

if __name__ == "__main__":
    #calc_g_G(2.0,15,50)
    all_of_them()
