#!/usr/bin/env python
# coding: utf-8

from keras import models
from keras import layers
import math
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from sktime.datatypes._panel._convert import (
    from_multi_index_to_nested,
    from_multi_index_to_3d_numpy,
    from_nested_to_multi_index,
    from_nested_to_3d_numpy,
)
#from sktime.utils.data_io import make_multi_index_dataframe
from sktime.datasets import make_multi_index_dataframe
import sys

from utility import parse_config

import sklearn
print(sklearn.__version__)

import sktime
print(sktime.__version__)


# initial filtering based on experimental setup
def initial_filtering(data):
    data = data[~data["comment"].isin(["stress_control"])]
    data = data[~data["comment"].isin(["H2B"])]
    data = data[data["guide"].str.contains('1398') | data["guide"].str.contains('1514')]
    data = data[data["time"] < 40]

    return data



def create_instance_index(data):
    # combine file and particle columns for using as instance index later on
    data['fp'] = data['file'] + '__' + data['particle'].astype(str)
    return data


def select_columns(data, cols):
    data = data[cols]
    return data


def nested_max(row, col_name='col'):
    return row[col_name].max()


def format_class_col(datan):
    datan['class'] = datan.apply(nested_max, axis=1, col_name='serum_conc_percent')
    datan['class'] = (datan['class'] / 10).astype('int')

    datan = datan.drop(columns=['serum_conc_percent'])
    return datan


def add_nframes_col(data):
    data = data.copy()
    data['nframes'] = data.groupby('fp')['frame'].transform('count')    
    return data


def fix_unequal_frame_counts(datan, n):
    # drop all series where frame count is not n
    datan = datan[datan['nframes']==n]
    return datan
    

def separate_observations_and_classes(datan):
    # separate class vector...
    y = datan['class'].values
    datan = datan.drop(columns=['class'])
    #print(datan.head())

    # ... and observations
    X = from_nested_to_3d_numpy(datan)
    return X,y


def displacement(row):
    return math.sqrt(row['dx']**2 + row['dy']**2)

def angle(row):
    return math.atan(row['dy'] / row['dx'])

def add_features(df):
    df = df.copy()

    # displacement
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dxy'] = df.apply(displacement, axis=1)
    df = df[df['frame']!=0]
    #df = df.reset_index()

    # direction
    df['angle'] = df.apply(angle, axis=1)
    df['dangle'] = df['angle'].diff()

    # changes in area and perimeter
    df['dperimeter'] = df['perimeter_au'].diff()
    df['darea'] = df['area_micron'].diff()
    df = df[df['frame']!=1]
  
    return df
    

# feature sets
# file is included in the first set because it is needed to create groups later on
fsets = {}
fsets['f_mot'] = ['x','y','min_dist_pxs','serum_conc_percent','file']
fsets['f_mot_morph'] = fsets['f_mot'] + ['area_micron','perimeter_au']
fsets['f_mot_morph_dyn'] = fsets['f_mot'] + ['dxy','angle','dangle','darea','dperimeter']
fsets['f_mot_morph_dyn_2'] = fsets['f_mot_morph'] + ['dxy','angle','dangle','darea','dperimeter']
fsets['f_x'] = ['x','serum_conc_percent','file']
fsets['f_y'] = ['y','serum_conc_percent','file']
fsets['f_xy'] = ['x','y','serum_conc_percent','file']
fsets['f_mindist'] = ['min_dist_pxs','serum_conc_percent','file']
fsets['f_morph'] = ['area_micron', 'perimeter_au','serum_conc_percent','file']
fsets['f_area'] = ['area_micron','serum_conc_percent','file']
fsets['f_perim'] = ['perimeter_au','serum_conc_percent','file']

# Discussion with Taras on 20220930: drop x and y, dxy is enough
fsets['f_dxy'] = ['dxy','serum_conc_percent','file']
fsets['f_dxy_angle'] = fsets['f_dxy'] + ['angle','dangle']
fsets['f_dxy_angle_morph'] = fsets['f_dxy_angle'] + ['area_micron','perimeter_au']
fsets['f_dxy_angle_morphd'] = fsets['f_dxy_angle_morph'] + ['darea','dperimeter']

# forgot mindist...
fsets['f_dxy_mindist'] = fsets['f_dxy'] + ['min_dist_pxs']
fsets['f_dxy_mindist_angle'] = fsets['f_dxy_mindist'] + ['angle','dangle']
fsets['f_dxy_mindist_angle_morph'] = fsets['f_dxy_mindist_angle'] + ['area_micron','perimeter_au']

# try also without perimeter (might be redundant with area)"
fsets['f_dxy_angle_area'] = fsets['f_dxy_angle'] + ['area_micron']
fsets['f_dxy_mindist_angle_area'] = fsets['f_dxy_mindist_angle'] + ['area_micron']

#print(fsets['f_mot_morph_dyn'])


def get_X_dfX_y_groups(data, f_set_name):
    data = data.copy()
    data = initial_filtering(data)
    data = create_instance_index(data)
    data = add_nframes_col(data)
    debug0 = data.copy()
    
    #print(data.nframes.unique())
    #print(data.groupby(by=['nframes']).count())
    # keep rows that have the maximum number of frames
    idxmax = data.groupby(by=['nframes']).count()['file'].idxmax()
    data = fix_unequal_frame_counts(data, idxmax)
    #print(data.nframes.unique())

    datam = data.set_index(['fp','frame'])
    datam.replace(to_replace=pd.NA, value=None, inplace=True)

    cols = fsets[f_set_name]
    datam = datam[cols]
    debug1 = datam.copy()

    datan = from_multi_index_to_nested(datam, instance_index='fp')
    debug2 = datan.copy()
    
    #print(datan['file'])
    # read group name from the last element of the series
    # index of first element might be 0,1 or 2, depending on how many elements
    # have been dropped because of adding features with df.diff()
    print(datan.columns)
    groups = datan['file'].apply(lambda x: x.iloc[-1])
    datan = datan.drop(columns=['file'])
    
    datan = format_class_col(datan)
    dfX = datan.drop(columns=['class'])

    X,y = separate_observations_and_classes(datan)
    return X, dfX, y, groups, debug1, debug2


import click

from utility import parse_config, set_logger

@click.command()
#@click.argument("config_file", type=str, default="config.yml")
@click.option("--config_paths", type=str, default="paths.yml")
@click.option("--config_tsc", type=str, default="config.yml")
def etl(config_paths, config_tsc):
    paths = parse_config(config_paths)
    config = parse_config(config_tsc)

    log_dir = paths["log"]["tsc"]
    
    # configure logger
    logger = set_logger(log_dir + "/etl_tsc.log")

    # Load config from config file
    logger.info(f"Load config from {config_tsc}")


    ''' 
    read the data 
    '''
    data_dir = paths['data']
    raw_data_file = config["etl"]["raw_data_file"]
    processed_data = config["etl"]["processed_data"]

    data = pd.read_csv(Path(data_dir) / raw_data_file)
    logger.info('Original data shape: ' + str(data.shape))
    data = add_features(data)
    datan = process_data(data)
    logger.info('Processed data shape (nested): ' + str(datan.shape))


    '''
    write processed data
    '''
    #datan.to_excel(Path(data_dir) / (processed_data + '.xlsx'), index=False)
    #datan.to_hdf(Path(data_dir) / (processed_data + '.h5'), index=False, key='tsc')
    datan.to_csv(Path(data_dir) / (processed_data + '.csv'), index=False)
        #X, dfX, y, groups, debugm, debugn = prepare_Xy(data, 'f_mot')


def load_data(path):
    data = pd.read_csv(path)
    data = add_features(data)
    return data
    
def load_df(hdf5='/proj/hajaalin/Projects/n_track_ML/data/63455ea_data_chromatin_live_nested.h5'):
    df = pd.read_hdf(hdf5)
    logger.info('read hdf5: ' + str(df.shape)) 


def load_csv(csv='/proj/hajaalin/Projects/n_track_ML/data/63455ea_data_chromatin_live_nested.csv'):
    df = pd.read_csv(csv)
    print(df.shape)
    
    
if __name__ == "__main__":
#    etl()
    load_csv()
