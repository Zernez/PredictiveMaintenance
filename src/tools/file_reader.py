import pickle
from pathlib import Path
import pandas as pd
from utility.event import Event
import config as cfg

class FileReader:

    def __init__ (self, dataset, type):
        if (dataset == "xjtu"):
            self.dataset_path = cfg.DATASET_PATH_XJTU + f'{type}/'
        elif (dataset == "pronostia"):
            self.dataset_path = cfg.DATASET_PATH_PRONOSTIA
        self.dataset= dataset


    def read_data_kaggle(self):
        """
        Deprecated Kaggle data extractor

        Returns:
        - set1: DataFrame containing data from 'set1_timefeatures.csv'
        - set2: DataFrame containing data from 'set2_timefeatures.csv'
        - set3: DataFrame containing data from 'set3_timefeatures.csv'
        """
        set1 = pd.read_csv("src/dataset/set1_timefeatures.csv")
        set2 = pd.read_csv("src/dataset/set2_timefeatures.csv")
        set3 = pd.read_csv("src/dataset/set3_timefeatures.csv")
        set1 = set1.rename(columns={'Unnamed: 0':'time'})
        set1.set_index('time')
        set2 = set2.rename(columns={'Unnamed: 0':'time'})
        set2.set_index('time')
        set3 = set3.rename(columns={'Unnamed: 0':'time'})
        set3.set_index('time')

        return [set1, set2, set3]
    
    def read_data(self, 
            test_condition: int, 
            bootstrap: int, 
            from_pickle: bool = False
        ) -> (pd.DataFrame, pd.DataFrame, dict):

        """
        Read the timeseries data from CSV files and optionally load events from pickle files.

        Args:
        - test_condition (int): The cardinal number of the test condition starting from 0.
        - bootstrap (int): The multiplier of the bootstrap.
        - from_pickle (bool, optional): Flag indicating whether to load events from pickle files. Defaults to False.

        Returns:
        - set_covariates (pandas.DataFrame): The covariates data.
        - set_boot (pandas.DataFrame): The boot data.
        - info_pack (dict): A dictionary containing the events data with keys 'KL' and 'SD'.
        """

        # Read the timeseries time and frequency data from the csv files
        set_covariates = pd.read_csv(self.dataset_path + 'covariates_' + str(test_condition) + '.csv')
        set_analytic = pd.read_csv(self.dataset_path + 'analytic_' + str(test_condition) + '.csv')
        if bootstrap > 0:
            set_boot = pd.read_csv(self.dataset_path + 'boot_' + str(test_condition) + '.csv')
        else:
            set_boot = []

        # Eventually read the events from the pickle files
        if from_pickle:
            event_kl = self.read_pickle(self.dataset_path + "event_kl")
            event_sd = self.read_pickle(self.dataset_path + "event_sd")
        else:
            event_kl, event_sd = Event(self.dataset, bootstrap).make_events(set_analytic, test_condition)

        # Pack the events in a dictionary
        info_pack = {'KL': event_kl, 'SD': event_sd}

        return set_covariates, set_boot, info_pack

    def read_pickle (self, 
        path: str
        ):

        file_handler = open(path, 'rb')
        obj = pickle.load(file_handler)
        file_handler.close()
        
        return obj
    