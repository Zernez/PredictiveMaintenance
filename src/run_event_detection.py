import numpy as np
import pandas as pd
import warnings
import config as cfg
from tools.file_reader import FileReader
from utility.builder import Builder
from utility.event import Event

warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

np.random.seed(0)

NEW_DATASET = False

def main():
    DATASET = "xjtu"
    N_BOOT = 0
    DATASET_PATH = cfg.DATASET_PATH_XJTU
    N_CONDITION = len(cfg.RAW_DATA_PATH_XJTU)
    
    # Create classes
    builder = Builder(DATASET, N_BOOT)
    event_detector = Event(DATASET)
    file_reader = FileReader(DATASET, DATASET_PATH)
    
    # For the first time running, a NEW_DATASET is needed
    if NEW_DATASET == True:
        builder.build_new_dataset(bootstrap=N_BOOT)
    
    # Extract information from the dataset selected from the config file
    model_results = pd.DataFrame()
    for test_condition in range (0, N_CONDITION):
        covariates, analytic = file_reader.read_data(test_condition, N_BOOT)
        kl_events, end_of_life = event_detector.compute_event_times(analytic, test_condition)
        
        bearing_idx = 0
        bearing_eol = end_of_life[bearing_idx]
        event_time, censored = event_detector.get_event_time(bearing_idx, kl_events, bearing_eol)
        
        print(0)
        
if __name__ == "__main__":
    main()