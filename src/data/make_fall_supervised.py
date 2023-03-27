from tools import file_writer, file_reader, preprocessor
from utility.config import load_config
import config as cfg
from pathlib import Path
import pandas as pd
import numpy as np

class MakeFallSupervised:

    def __init__(self):
        pass
    
    file_reader = file_reader.FileReader()
    preprocessor = preprocessor.Preprocessor()
    file_writer = file_writer.FileWriter()

    # Load data
    df = file_reader.read_csv(Path.joinpath(cfg.PROCESSED_DATA_DIR, 'home_care_ts.csv'))

    # Load seetings
    settings = load_config(cfg.CONFIGS_DIR, 'data.yaml')
    window_len = settings['window_len']
    lag = settings['lag']

    # Define columns
    label_cols = ['Id', 'Time', 'Period', 'Weeks', 'Observed']
    hc_cols  = list(df.columns.drop(label_cols + ['BirthYear', 'Gender']))

    # Only include citizens which initial hc is over 200
    obs_mask = df.groupby('Id').first()[hc_cols].sum(axis=1) > 200
    pos_citizens = list(obs_mask[obs_mask].index)
    df = df[df['Id'].isin(pos_citizens)].reset_index(drop=True)

    # Only include citizens between 30-50 birth year
    df = df.loc[(df['BirthYear'] >= 30) & (df['BirthYear'] <= 50)]
    df = df.drop(['BirthYear', 'Gender'], axis=1)

    # Make moving average of HC columns
    df_ma = preprocessor.series_to_moving_average(df, window_len=window_len,
                                                lag=lag, lbl_cols=label_cols)

    # Drop id, time, period columns
    df_ma = df_ma.drop(['Id', 'Time', 'Period'], axis=1)

    # Save files
    file_writer.write_csv(Path.joinpath(cfg.PROCESSED_DATA_DIR, 'home_care_ma.csv'), df_ma)