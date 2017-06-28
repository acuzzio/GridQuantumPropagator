''' This script launches a single Point propagation from a h5 file. '''

from collections import namedtuple
from argparse import ArgumentParser
from quantumpropagator import single_point_propagation, printEvenergy, specificPulse


def read_single_arguments(single_inputs):
    '''
    This funcion reads the command line arguments and assign the values on
    the namedTuple for the 0D grid propagator.
    '''
    parser = ArgumentParser()
    parser.add_argument("-n", "--output",
                        dest="n",
                        type=str,
                        help="output folder")
    parser.add_argument("-s", "--totalSteps",
                        dest="s",
                        type=int,
                        help="Total time of propagation")
    parser.add_argument("-d", "--deltaTime",
                        dest="d",
                        type=float,
                        help="Time step of propagation")
    parser.add_argument("-i", "--h5file",
                        dest="i",
                        type=str,
                        required=True,
                        help="H5 input file")
    parser.add_argument("-e", "--energy",
                        dest="e",
                        action='store_true',
                        help="Get Ev energies and TDM from H5 file.")

    args = parser.parse_args()

    if args.n != None:
        single_inputs = single_inputs._replace(out_folder=args.n)
    if args.s != None:
        single_inputs = single_inputs._replace(nsteps=args.s)
    if args.d != None:
        single_inputs = single_inputs._replace(dt=args.d)
    if args.i != None:
        single_inputs = single_inputs._replace(H5file=args.i)
    if args.e != None:
        single_inputs = single_inputs._replace(Energies=args.e)
    return single_inputs

single_inputs = namedtuple("single_input",
            ("out_folder",
             "nsteps",
             "dt",
             "H5file",
             "Energies",
             "graphs",
             "outF"
            )
            )

def main():
    '''
    Launches a single Point propagation from a h5 file.
    '''
    inputs = single_inputs(".",              # out_folder
                           5,                # nsteps
                           0.04,             # dt
                           "nothing",        # H5file
                           False,            # Energies
                           True,            # graphs
                           True              # outF
                           )
    new_inp = read_single_arguments(inputs)
    if new_inp.Energies == True:
       printEvenergy(new_inp.H5file)
    else:
       single_point_propagation(new_inp.H5file, new_inp.dt, new_inp.nsteps, specificPulse,
            'CisButa', new_inp.graphs, new_inp.outF, new_inp.out_folder)

if __name__ == "__main__":
    main()

