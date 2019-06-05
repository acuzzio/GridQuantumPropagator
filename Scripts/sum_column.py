#!/home/alessio/config/miniconda/envs/quantumpropagator/bin/python
'''
This is to sum up Absorbing potential lines
'''

from argparse import ArgumentParser
import os
import pandas as pd

def read_single_arguments():
    '''
    Command line reader
    '''
    description_string = 'This script will launch a Grid quantum propagation'
    parser = ArgumentParser(description=description_string)
    parser.add_argument("-t", "--time",
                        dest="t",
                        type=float,
                        help="This is the time at which you want to cut")
    return parser.parse_args()


def main():
    '''
    The main function
    '''
    args = read_single_arguments()
    file_name = 'Output_Abs'
    if os.path.isfile(file_name):
        data_frame = pd.read_csv('Output_Abs', header=None, delim_whitespace=True,
                                 names=['fs', 'abs_tot', 'abs_0', 'abs_1', 'abs_2',
                                        'abs_3', 'abs_4', 'abs_5', 'abs_6', 'abs_7'])
        print('\nLast Point is {} AU\n'.format(data_frame['fs'].iloc[-1]))
        print('\nColumn sum of absorbing potential is:\n')
        delta_t = data_frame['fs'][1]
        print(data_frame.sum(axis=0))
        print('\nMultiplied by dt\n')
        print((data_frame*delta_t).sum(axis=0))

        if args.t != None:
            print('\nI will cut and sum at {} AU\n'.format(args.t))
            smaller_data_frame = data_frame[data_frame['fs'] < args.t]
            print(smaller_data_frame.sum(axis=0))
            print('\nMultiplied by dt\n')
            print((smaller_data_frame*delta_t).sum(axis=0))
    else:
        print('in this folder there is no {} file'.format(file_name))


if __name__ == "__main__":
    main()
