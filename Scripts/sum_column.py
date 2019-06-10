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
    file_name2 = 'Output_Regions'
    if os.path.isfile(file_name):
        data_frame = pd.read_csv(file_name, header=None, delim_whitespace=True,
                                 names=['fs', 'abs_tot', 'abs_0', 'abs_1', 'abs_2',
                                        'abs_3', 'abs_4', 'abs_5', 'abs_6', 'abs_7'])
        data_frame2 = pd.read_csv(file_name2, header=None, delim_whitespace=True,
                                  names=['FC', 'Reac', 'Prod'])
        len_df_2 = len(data_frame2.index)
        print('\nLast Point is {} AU\n'.format(data_frame['fs'].iloc[-1]))
        print('\nColumn sum of absorbing potential is:\n')
        delta_t = data_frame['fs'][1]
        print(data_frame.sum(axis=0))
        print('\nMultiplied by dt\n')
        print((data_frame*delta_t).sum(axis=0))

        if args.t != None:
            smaller_data_frame = data_frame[data_frame['fs'] < args.t]
            length_at_cut = len(smaller_data_frame.index)
            print('\nI will cut and sum at {} AU. # {} lines.\n'.format(args.t, length_at_cut))
            print(len(smaller_data_frame))
            print(smaller_data_frame.sum(axis=0))
            print('\nMultiplied by dt\n')
            print((smaller_data_frame*delta_t).sum(axis=0))
            if length_at_cut < len_df_2:
                value_of_region_at_cut = data_frame2.loc[length_at_cut]
                print('\nRegion file at same point:\n{}\n'.format(value_of_region_at_cut))
            else:
                print('not enough points in region file.')
    else:
        print('in this folder there is no {} file'.format(file_name))


if __name__ == "__main__":
    main()
