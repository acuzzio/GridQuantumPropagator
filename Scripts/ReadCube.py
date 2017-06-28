''' This script reads and make operations on cubes '''

from collections import namedtuple
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import os

def vmd_watch_cube(fn):
    '''
    This function will launch VMD to display a cube.
    '''
    cwd = os.getcwd()
    vmd_script = fn + "vmdscript"
    vmd_content = '''
mol new {{{}/{}}} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}
color Display Background white
display projection Orthographic
display depthcue off
axes location Off
mol modstyle 0 0 Licorice 0.100000 35.000000 35.000000
mol selection all
mol addrep 0
mol modstyle 1 0 Isosurface -0.000200 0 0 0 1 1
mol modcolor 1 0 ColorID 0
mol representation Isosurface -0.000200 0 0 0 1 1
mol selection all
mol material Opaque
mol addrep 0
mol modstyle 2 0 Isosurface 0.000200 0 0 0 1 1
mol modcolor 2 0 ColorID 1
'''.format(cwd,fn)
    f = open(vmd_script,'w')
    f.write(vmd_content)
    f.close()
    os.system("vmd -e " + vmd_script)
    os.system("rm " + vmd_script)

def read_this_arguments(single_inputs):
    '''
    input reader for the cube script
    '''
    parser = ArgumentParser()
    parser.add_argument("-c", "--cube",
                        dest="c",
                        type=str,
                        help="cube file (cube)")
    parser.add_argument("-v", "--vmd",
                        dest="v",
                        type=str,
                        help="Visualize cube in VMD")

    args = parser.parse_args()

    if args.c != None:
        single_inputs = single_inputs._replace(cube=args.c)
    if args.v != None:
        single_inputs = single_inputs._replace(vmd=args.v)

    return single_inputs

single_inputs = namedtuple("single_input",
            ("cube",
             "vmd"
            )
            )

def main():
    '''
    reads and makes operations on cubes
    '''
    inputs = single_inputs("Density_21.cube", # cube
                           "Empty_thing" # vmd
                           )

    new_inp = read_this_arguments(inputs)
    fn = new_inp.cube
    if new_inp.vmd != "Empty_thing":
        vmd_watch_cube(new_inp.vmd)
    else:
        lol = pd.read_table(fn, delim_whitespace=True,
                 skiprows=14).as_matrix().flatten()
        lol = lol[~np.isnan(lol)]  # take out nan values
        cell_dimension = 0.0005786967592870371 # PROBLEM BOUND I need to take
                                               # this number from the cube file !
        integral = np.sum(lol*cell_dimension)
        dimension = lol.shape[0]
        print('\nFile {}:\nGrid points: {}\nIntegral: {}'.format(new_inp.cube,
                                           dimension, integral))

if __name__ == "__main__":
    main()

