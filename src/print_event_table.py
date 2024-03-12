import pandas as pd
import config as cfg
import numpy as np
from tools.file_reader import FileReader
from utility.builder import Builder
from utility.event import EventManager
from utility.data import get_lmd

NEW_DATASET = False

if __name__ == "__main__":
    DATASET = "xjtu"
    N_BOOT = 0
    DATASET_PATH = cfg.DATASET_PATH_XJTU
    N_CONDITION = len(cfg.RAW_DATA_PATH_XJTU)
    
    if NEW_DATASET == True:
        Builder(DATASET, N_BOOT).build_new_dataset(bootstrap=N_BOOT)
    
    for test_condition in [0, 1, 2]:
        pct_error_list = list()
        _, analytic = FileReader(DATASET, DATASET_PATH).read_data(test_condition, N_BOOT)
        event_manager = EventManager(DATASET)
        event_times = event_manager.get_event_times(analytic, test_condition,
                                                    get_lmd(test_condition))
        failure_times = event_manager.get_eol_times(analytic)
        for bearing_id in range(1, 10):
            event_time = event_times[bearing_id-1]
            failure_time = failure_times[bearing_id-1]
            error = event_time - failure_time
            pct_error = ((event_time - failure_time)/ failure_time) * 100
            pct_error_list.append(pct_error)
            print(f"{event_time} & {failure_time} & {error} & {round(pct_error, 1)}")
        print()