"""
This file makes list both the list of citizens and the different dictionaries to be used in
"""

from tools import alarm_labeler, file_writer, file_reader, alarm_inputter
from utility import data
from utility.config import load_config
import config as cfg
from pathlib import Path
import pandas as pd
import numpy as np
from time import time

class MakeFallTimeseries:

    def __init__(self):
        pass

    file_reader = file_reader.FileReader()
    alarm_labeler = alarm_labeler.AlarmLabeler()
    file_writer = file_writer.FileWriter()
    alarm_inputter = alarm_inputter.AlarmInputter()
    data = data.Data()

    # Load data
    ats = file_reader.read_pickle(Path.joinpath(cfg.INTERIM_DATA_DIR, 'ats.pkl'))
    hc = file_reader.read_pickle(Path.joinpath(cfg.INTERIM_DATA_DIR, 'home_care.pkl'))

    # Load settings
    settings = load_config(cfg.CONFIGS_DIR, 'data.yaml')
    alarm_ats = settings['alarm_ats']
    dropout_threshold = settings['dropout_threshold']

    # Clean the ats dataset further
    ats['YearWeek'] = pd.to_datetime(ats['LendDate']).dt.strftime('%Y-%V')
    ats[['Year', 'Week']] = ats['YearWeek'].str.split('-', 1, expand=True)
    ats['Week'] = ats['Week'].str.lstrip('0')
    ats = ats.drop('YearWeek', axis=1)
    ats = ats.replace(r'^\s*$', np.nan, regex=True).dropna()
    ats['Year'] = ats['Year'].astype(int)
    ats['Week'] = ats['Week'].astype(int)

    # Gather data on alarms, split into alarms and other ats
    is_alarm_lend = ats.apply(lambda x: 1 if alarm_ats in x['DevISOClass'] else 0, axis=1)
    alarms = ats.loc[is_alarm_lend == 1][['CitizenId', 'LendDate', 'Year', 'Week']]
    alarms_before = alarms.loc[alarms['LendDate'] < '2022-01-01']
    alarms_after = alarms.loc[alarms['LendDate'] >= '2022-01-01']
    ats = ats.loc[ats['DevISOClass'] != alarm_ats] # remove alarm

    # Make alarm splits
    alarm_split = alarms_after.loc[(alarms_after['Year'] == 2022) & (alarms_after["Week"] <= 26)]
    alarms = [alarm_split]

    # Make home care splits
    hc = hc.reset_index(drop=True)
    hc['Year'] = hc['Year'].astype(int)
    hc['Week'] = hc['Week'].astype(int)
    hc_split = hc.loc[(hc['Year'] == 2022) & (hc["Week"] <= 26)]
    hc = [hc_split]
    all_hc = pd.concat([hc_split], axis=0)
    care_dict = data.make_type_dict(all_hc['CareType'].unique())
    care_types = list(care_dict.keys())

    all_data = pd.DataFrame()
    evaled_citizens = list(alarms_before['CitizenId'])
    
    for i, (hc_df, alarm_df) in enumerate(zip(hc, alarms)):
        start_time = time()
        # Remove citizens that have already been evaluated (alarm/dropout/fall)
        hc_df = hc_df[~hc_df['CitizenId'].isin(evaled_citizens)].reset_index(drop=True)
        alarm_df = alarm_df[~alarm_df['CitizenId'].isin(evaled_citizens)].reset_index(drop=True)

        # Convert data
        general_df = hc_df[['CitizenId','BirthYear','Gender']].groupby('CitizenId').first().reset_index()
        alarm_df = alarm_df[['CitizenId','Year','Week']].to_numpy()

        # Collect features and labels
        citizen_ids = hc_df['CitizenId'].unique()
        citizen_id_to_id = {citizen_ids[i]: i for i in range(len(citizen_ids))}
        dates, date_dict = data.make_date_dict(hc_df)
        hc_features = alarm_inputter.make_hc_features(hc_df, citizen_id_to_id,
                                                    dates, date_dict, care_dict)
        start_at_ts = alarm_labeler.get_starts(hc_features, len(date_dict))
        alarm_at_ts = alarm_labeler.get_alarms(alarm_df, citizen_ids, date_dict)
        dropout_at_ts = alarm_labeler.get_dropouts(hc_features, start_at_ts,
                                                citizen_ids, dropout_threshold)

        # Save citizens that get alarm/dropout/fall in this period
        evaled_citizens = evaled_citizens \
            + list(citizen_ids[np.where(alarm_at_ts < np.inf)]) \
            + list(citizen_ids[np.where(dropout_at_ts < np.inf)])

        # Turn the care array into a dataframe
        df_index = pd.MultiIndex.from_product([range(s) for s in hc_features.shape])
        df = pd.DataFrame({'HC': hc_features.flatten()}, index=df_index)['HC']
        df = df.unstack().swaplevel().sort_index()
        df.columns = care_types
        df.index.names = ['Time', 'Id']
        df = df.reset_index()

        # Make alarm label based on weeks/observed
        df = alarm_labeler.make_alarm_label(df, start_at_ts, alarm_at_ts, dropout_at_ts)

        # Add general features
        ids = general_df['CitizenId'].unique()
        mapping = {ids[i]: i for i in range(len(ids))}
        general_df['CitizenId'] = general_df['CitizenId'].replace(to_replace=mapping)
        df = pd.merge(df, general_df, left_on="Id", right_on="CitizenId").drop('CitizenId', axis=1)

        # Merge dataframes
        df['Period'] = i
        all_data = pd.concat([all_data, df], axis=0, ignore_index=True)
        passed_time = time() - start_time
        print(f"Split {i} took {passed_time} with {len(citizen_ids)} citizens")

    # Zero index the id
    citizen_ids = all_data['Id'].unique()
    mapping = {citizen_ids[i]: i for i in range(len(citizen_ids))}
    all_data['Id'] = all_data['Id'].replace(to_replace=mapping)

    # Arrange columns and concert types
    lbl_cols = ['Id', 'Time', 'Period', 'Weeks', 'Observed']
    df_ts = all_data[lbl_cols + list(all_data.columns.drop(lbl_cols))]
    df_ts = df_ts.convert_dtypes()

    # Save file
    file_writer.write_csv(Path.joinpath(cfg.PROCESSED_DATA_DIR, 'home_care_ts.csv'), df_ts)
