import numpy as np
import pandas as pd
import warnings
import config as cfg
from tools.file_reader import FileReader
from utility.builder import Builder
from utility.event import EventManager

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)

NEW_DATASET = False

def main():
    DATASET = "xjtu"
    N_BOOT = 0
    DATASET_PATH = cfg.DATASET_PATH_XJTU
    N_CONDITION = len(cfg.RAW_DATA_PATH_XJTU)
    
    # For the first time running, a NEW_DATASET is needed
    if NEW_DATASET == True:
        Builder(DATASET, N_BOOT).build_new_dataset(bootstrap=N_BOOT)
    
    # Extract information from the dataset selected from the config file
    for test_condition in range(0, N_CONDITION):
        _, analytic = FileReader(DATASET, DATASET_PATH).read_data(test_condition, N_BOOT)
        event_times = EventManager(DATASET).get_event_times(analytic, test_condition)
        print(event_times)
        
if __name__ == "__main__":
    main()