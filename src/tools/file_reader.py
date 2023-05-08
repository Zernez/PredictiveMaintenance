import pickle
from pathlib import Path
import pandas as pd

class FileReader:

    def __init__(self):
        pass

    def read_data_kaggle(self):
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
    
    def read_data_xjtu(self):
        dataset_path = "./data/XJTU-SY/csv/"
        set_covariates= pd.read_csv(dataset_path + 'covariates.csv')
        set_boot= pd.read_csv(dataset_path + 'boot.csv')

        event_kl= self.read_pickle("./data/XJTU-SY/csv/" + "event_kl")
        event_sd= self.read_pickle("./data/XJTU-SY/csv/" + "event_sd")
        info_pack= {'KL': event_kl, 'SD': event_sd}

        return set_covariates, set_boot, info_pack



    def read_pickle(self, path: Path):
        """
        Loads the pickled object at the location given.

        :param path: Path (including the file itself)
        :return: obj
        """
        file_handler = open(path, 'rb')
        obj = pickle.load(file_handler)
        file_handler.close()
        return obj
    